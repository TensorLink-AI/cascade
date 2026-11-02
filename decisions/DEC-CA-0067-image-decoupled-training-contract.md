---
id: DEC-CA-0067
type: decision
title: "Worker image decoupled from the training contract, plus the v2 verdict budget (stake-majority vote)"
status: accepted
date: 2026-11-01
tags: [consensus, contract, activation, worker-image, eval]
relations: {amends: [DEC-CA-0036, DEC-CA-0050, DEC-CA-0055, DEC-CA-0066], uses: [DEC-CA-0045, DEC-CA-0048, DEC-CA-0049]}
---

## Problem

`[v2.training] train_image_digest` (the worker image pin) is a field of
`TrainingContractConfig`, so it is folded into `contract_digest`. Re-pinning the
image is therefore a contract change:

- The trainer's `_stale_contract` and settlement drop the carried king entry and
  the finished legs, then retrain them. The operator pays for those retrains.
- Validators reject any entry whose `training_contract_digest` differs from the
  manifest's `contract_digest` (`era_entry_init_mismatch`). They also hold one
  king pointer per v2 seed epoch, so a retrained king gets
  `era_king_pointer_mismatch`.

The result is that an image repin inside a seed epoch deadlocks every settlement
on every validator. Image upgrades (speed, an EMA shadow) must ship at any era
boundary. The numerics they train under must stay enforced.

## Rule

`numerics_contract_digest(contract)` is
`sha256("cascade-numerics-contract/1\n" + canonical JSON)`. The JSON is the
`contract_payload` that `contract_digest` hashes, with `IMAGE_RUNTIME_FIELDS =
("train_image_digest", "continuation_protocol")` removed. The domain tag means it can never equal a full
digest. `contract_digest` itself is unchanged, so every existing digest is
byte-identical (golden vector).

An audit of `TrainingContractConfig` found two runtime-only terms: the image pin
and `continuation_protocol`.

`continuation_protocol` turns on optimizer-boundary snapshots and resume. It
cannot change the trained weights:
- `cascade/trainer/continuation.py` (module docstring): "Input iterators are
  reconstructed by deterministic replay before RNG restoration. A continuation
  is never a new warm start and never increases the remaining budget."
- docs/OPERATOR.md "Qualified training continuation": recovery replays completed
  batches without optimizer updates and checks their digest before restoring
  the saved RNG state; snapshot I/O adds no optimizer steps or budget.
- The qualification run was bitwise: OPSLOG 2026-11-01 08:50, rc15 vs rc16
  bit-identical, and resume bitwise including the EMA.

Its one other effect is the shard stream's eager accounting
(`cascade/trainer/stream.py` `_ShardStream`). The same series are yielded in the
same order, and the final consumed series is also counted in the receipt's
`corpus_digest` / `n_series`. That is accounting, not numerics. It is carried by
each leg's own receipt; validators do not recompute it.

The proof tries both protocol values automatically. Enabling continuation after
the switch therefore keeps the king and the legs, exactly like an image repin.

For the other terms: the schedules and `router_slow` are already outside the digest. `expected_gpu`
stays because it is a LOCKED term. `max_train_seconds` stays because the wall
can cut the tokens a leg trains.

The rule applies to every settlement of an era starting at or after `[v2]
image_decoupled_from_block`. The settlement at the block itself is the previous
era's last settlement, so no era is judged under two rules.

- **Trainer.** It stamps `numerics_contract_digest` on the manifest and on every
  entry. Both are SIGNED and drop-when-empty, and are never emitted before the
  switch. A leg stays fresh when its numerics are proven equal to the era's.
- **Proof.** A leg records only its full digest. It trained under the current
  contract with only the image changed if and only if, for some image `img`,
  `contract_digest(current contract with train_image_digest = img)` equals that
  digest. The images tried are the current pin and every
  `[v2] prior_train_image_digests` entry (`prove_numerics_digest`). A leg that
  cannot be proven is stale and retrains, with an ERROR log.
- **Validator** (`era_gate`). The manifest's numerics digest must exist and equal
  the numerics of its own `contract_body`, or of the local contract when there is
  no body. Every entry's numerics digest must exist and equal it
  (`era_numerics_missing|mismatch`, `era_entry_numerics_missing|mismatch`). The
  per-entry init check and the LOCKED-term gate are unchanged. The king pointer
  rule is unchanged: the king is no longer retrained, so its pointer is kept.
- **Unchanged.** The full declared image is still recorded in `contract_body`
  and in each entry's `training_contract_digest`. Pod-side verification is
  untouched: the worker still refuses a FINAL whose runtime image differs from
  the pin, and the provision codeprint is unchanged.
- **Bundle.** From the same block, a v2 settlement's private verdict budget is
  `[v2] eval_n_windows_after`, using the explicit-evidence verdict draw. Lane A,
  lane B, the trainer's lane-B self-judge and the audit replay all use it.
  Mainnet ships 4000; testnet ships 0, so it keeps its small budget.
  - Finding: the mainnet `eval_n_windows = 4000` that DEC-CA-0066 describes sits
    under `[round]`, not `[v2]`. No parser reads it there, so v2 settlements
    judge on `[eval] n_windows`.

## Activation: stake majority, then the next era start

This is the DEC-CA-0045 secondary-feature path, aligned to eras like
DEC-CA-0049.

- **Feature.** The name is
  `v2b-<sha256("image-decoupled/1|train_image_digest,continuation_protocol|eval_n_windows_after=<n>")[:8]>`
  (mainnet: `v2b-150ec0d0`). It is short because of the note budget below.
  A changed bundle is a fresh vote.
- **Signal.** Validators on this release add the segment to their note at startup
  whenever `[activation]` is on and v2 is scheduled.
- **Lock-in.** It locks in at the first boundary where `[activation] threshold` of
  eligible stake has signalled. The switch is the first ERA START strictly after
  that boundary.
- **Persistence.** The record lives in `activation_image_decoupled_state.json`
  and is one-way. A late node adopts the block from the notes
  (`agreed_activation`).
- **Owner override.** A typed `image_decoupled_from_block` wins.
- **Receipts.** Receipts stamp `image_decoupled_block` (drop-when-default) so the
  audit replays each settlement under its rule.
- **Trainer.** The trainer resolves the same feature read-only, with the same
  resolver, the same as-of-boundary reads and the same notes, and never signals.
  It cannot stamp early, because the block is 0 until it resolves. A lock-in
  always lands at least one settlement before its era starts.

## Blast radius / fork risk (state BEFORE deploying)

- **Old validators.** Old validators break after the switch.
  - `load_manifest` (`cascade/shared/manifest.py` ~L862 on main) builds entries
    from known keys only.
  - `verify_signature` (~L950) re-verifies over the re-serialised
    `canonical_body()`, not the raw text.
  - An old parser therefore drops the new keys, its recomputed body no longer
    matches the trainer's signature, and it rejects every post-switch settlement
    as `signature_invalid` (`cascade/validator/gating.py` L43).
  - Before the switch nothing new is emitted, so manifests and signatures are
    byte-identical and old and new validators agree.
  - Old validators parse the extra note segment generically: `parse_signals`
    accepts any well-formed segment.
- **Coordination.** The vote IS the coordination: validators holding the
  threshold stake must run the release. Any validator still on the old release at
  the switch forks off (all its v2 settlements are rejected) until it upgrades.
  With six external validators on netuid 91, confirm every one has upgraded
  (their notes carry the `v2b-` segment) before relying on the lock-in. Test on
  testnet first.
- **Note byte budget, and the typed blocks that free it.** The plain commitment
  is one 128-byte field. Before this release:
  - The main validator, whose rollover is typed in its deployed chain.toml,
    publishes
    `cascade-ready:1:rolling-era-king:0:0:forfeit-42b1a486:9165600:9166500:margin-v2-d30eb443:9190800:9194400:prelaunch-95-r2:0:0`,
    which is 124 bytes. Adding `:v2b-<8hex>:0:0` (+17 bytes) makes 141.
  - A validator on the repo chain.toml (rollover resolved from signals) already
    sits at 136 bytes. Its trimmer drops `prelaunch-95-r2` today, and would drop
    `v2b-` too.
  - Either way the vote could not be signalled.

  This release therefore types the two already-decided blocks into
  `chain.toml [scoring]`:
  - `forfeit_from_block = 9166500`: the `forfeit-42b1a486` lock-in (9165600)
    resolved through `forfeit_block_for`, which returns the activation block
    unchanged. The deployed validator log confirms it: "forfeiture ARMED at
    block 9166500".
  - `margin_v2_from_block = 9194400`: the `margin-v2-d30eb443` lock-in (9190800)
    resolved through `apply_margin_v2_activation`, which writes the era-start
    activation block. The log confirms "bar v2 ARMED at block 9194400".

  The typed values equal the resolved ones exactly, so judged behaviour is
  identical:
  - A persisted lock-in restored on the typed config is a no-op, never a raise,
    because `apply_*` returns a typed config unchanged.
  - `resolve_activation` re-labels the record `source = "config"`.
  - A typed feature is not `decided_on_chain`, so `own_signal_payload` omits its
    segment.
  - Receipts stop stamping `margin_v2_block`, which is now typed. Archived
    receipts that carry 9194400 still pass `check_margin_v2` against the typed
    value.
  - `prelaunch-95-r2` is left alone. It becomes signalled again by
    repo-config validators, but v2 has launched and every future settlement is
    a v2 settlement, so a pre-launch burn lock-in is inert
    (`cascade.shared.emission`).

  Note lengths after this release:

  | Validator | `v2b-` unlocked | after lock-in (`v2b-<8hex>:<lock>:<act>`) |
  |---|---|---|
  | repo-config validator | 85 bytes | 97 bytes |
  | main validator (typed rollover) | 73 bytes | 85 bytes |

  `test_mainnet_note_carries_v2b_within_the_field` asserts at least 24 bytes of
  headroom.
- **What every validator operator must do.**
  - External validators run `cascade-validator --chain-toml chain.toml` from
    their repo checkout (docs/VALIDATOR.md: "the published file is the
    authority for every consensus value"). They must deploy THIS release's
    `chain.toml`, which carries the typed blocks, keeping their own local edits.
    A node still on the old file keeps signalling the forfeit and margin-v2
    segments, its trimmer drops `v2b-`, and its stake never counts toward the
    vote.
  - The owner overlay (DEC-CA-0057) cannot do this for them. An overlay that
    changes a `*_from_block` to a block before its own apply block is refused
    as retroactive.
  - The main validator's deployed chain.toml carries uncommitted live
    overrides, so the same two typed lines must be applied to it by hand. That
    edit is Approve-tier and is not part of this PR.

## Per-leg contract re-derivation (no trusted stamps)

From the switch every entry also publishes its own runtime terms,
`train_image_digest` and `continuation_protocol`. Both are SIGNED and dropped
when empty, so pre-switch bytes are identical.

Validators (`era_gate`, `era_entry_contract_unproven`) and the audit
(`numerics-contract`) re-derive
`contract_digest(era contract body with the entry's image/protocol) == entry.training_contract_digest`,
and require the entry's numerics to equal the manifest's. A garbage or
mismatched digest is rejected, so the numerics stamp is never taken on trust.

Audit replay (`cascade/audit/rederive.py` `entry_replay_contract`) runs each
leg under its OWN recorded image and protocol:
- Tier 2 byte-exactness compares against the image the leg actually ran.
- Tier 1 re-derives the corpus digest with the leg's continuation protocol, so
  the eager-accounting `corpus_digest` / `n_series` difference never
  false-FAILs.

## Automatic image repin at the switch

The trainer-only `[v2] *_after` pins schedule the image change onto the same
block, so no hand repin is needed:
- `train_image_digest_after`, `funded_pod_image_after`,
  `worker_code_fingerprint_after`, and `funded_sku_wall_measured_image_after`
  while the wall gate is on.
- They are loaded as one release: all set or none, and the pod image and the
  training image must name the same digest.

How they apply:
- `train_image_digest_after` is bound onto the v2 contract as a schedule. It is
  never in any digest (`_NEVER_IN_DIGEST`). `TrainingContractConfig.at_block`
  of an era starting at or after the resolved switch pins it; earlier eras pin
  the current image.
- Every leg's contract therefore carries its era's image. The `[round]` pins
  for rents follow the leg's contract (`cascade.trainer.leg_pins.leg_round`).
- While the after pins are configured, a dispatcher hook does two things:
  - it checks each host's worker-code fingerprint against the leg's era pin, so
    a host on the other image never takes the leg;
  - for a post-switch leg, it pushes a deployed chain.toml with the after pins
    substituted, so a worker of any release (rc16 predates this change)
    resolves the same contract digest.
- The pre-switch pin is proven automatically.
- Nothing changes before the switch resolves. The validator side is untouched:
  the image is outside consensus from the same block.

The operator lines for rc16 are in docs/OPERATOR.md, "Automatic image repin at
the switch".

Lane handling:
- A lane pool picks a lane on the leg's era image (`_dispatch_on_free_lane`
  with the hook's `matches`); it does not take one and then fail.
- No matching lane means a clean `LegImageMismatch`, which the lane breaker
  never counts.
- The fingerprint cache is keyed on endpoint plus host key plus allocation, has
  a 300 s TTL, and is always re-measured before a refusal. A lane
  re-bootstrapped onto the after image is accepted without a trainer restart.
- The provisioner health gate also accepts the after release.
- `check_worker_release` accepts a box whose code is either release.
- The bench-anneal telemetry leg (`runner_promotion`) runs on the pod that
  trained the entry and now carries its duel's `--contract-block` (the era
  start), so the hook checks that pod's image like any other leg.

**Owner rule: qualification before arming.**
- An image may be set as `*_after` ONLY after a bit-identical qualification
  against the current image: weights and optimizer state at every world size
  the trainer can rent for a v2 leg, and crash/resume at each of them.
- The qualification must be recorded in OPSLOG before the `*_after` lines are
  armed.
- Excluding the image from the numerics digest is sound only because of this
  rule.

**The rule is ENFORCED by the trainer** (owner decision 2026-11-01):
- The trainer uses the `*_after` pins only when the repo-tracked record
  `qualifications/<from12>-to-<to12>.json` exists for EXACTLY the pair
  (current `[v2.training] train_image_digest` → `[v2] train_image_digest_after`)
  and validates (`cascade/shared/image_qualification.py`): schema
  `cascade-image-qualification/1`; full from/to image digests; from/to worker
  code fingerprints equal to the pinned `[round] worker_code_fingerprint` /
  `[v2] worker_code_fingerprint_after`; `date_utc`; ≥ 1 run id;
  `world_sizes` ⊇ the REQUIRED sizes; `resume_tested = true`; and per world
  size the sha256 of the sorted per-tensor comparison manifest (weights,
  stable weights, optimizer tensors) with `equal = true`, plus
  `resume_equal = true` at every required size.
- Required world sizes (owner 2026-11-02: "we don't use 1x GPU on training"):
  the GPU counts of the SKUs the trainer can rent for v2 legs, from the
  effective `[round] funded_pod_skus` plus a set `funded_pod_sku`
  (`image_qualification.required_worlds`; `"8xH200"` → 8, a plain type → 1).
  Mainnet's `["8xH200", "8xH100", "4xH200", "4xH100"]` requires {4, 8}; world
  1 is required only while a one-GPU SKU is listed. With no SKU configured
  the conservative {1, 8} applies. Adding a SKU with a new GPU count
  invalidates existing records until that size is qualified.
- `scripts/write_image_qualification.py` emits the record from the
  qualification harness outputs (`scripts/qualify_pyramid_bitwise.py`, A run
  in the FROM image, B in the TO image). It hashes tensors without torch and
  writes nothing unless the record validates.
- Refusal: no post-switch rent (`leg_pins.leg_round`), dispatch (the host
  hook), lane take (the lane pool refuses before taking a lane) or funded leg
  (before its corpus or rent) on the after image. The leg fails as an
  operator fault (`LegImageMismatch` ⊂ `_OperatorFault`), unburned, never
  counted by the lane breaker, and NEVER falls back to the old image: its
  contract pins the after image, so it waits for a record. The provisioner
  health gate admits after-image lanes and `check_worker_release` accepts the
  after release only when qualified.
- Signal: an ERROR every trainer tick and an operator-watchdog alert
  (`image_after_unqualified`, scope `image`) while the after pins are armed
  without a valid record — from arming, before the switch resolves.
- Emergency override: `[v2] image_after_unqualified_override = true`
  (trainer-only, default false). The pins are used unqualified, with an ERROR
  every tick and an `image_after_override` alert while it is on.
- Pre-switch eras and configs without after pins are unaffected.

## Release notes / blast radius (operators)

- **Typed forfeit / margin-v2 blocks.**
  - Late-joining or old-release validators can no longer re-derive the forfeit
    and margin-v2 blocks from notes. This release no longer signals them
    (typed), so such a node must run this release's chain.toml.
  - Receipts stop carrying `margin_v2_block` (the block is typed). This is
    audit-only: archived receipts with 9194400 still pass.
- **Note budget.** The `v2b-` segment fits beside a locked `cfg-` overlay
  segment while blocks have 7 digits. Past block 10,000,000 a locked `v2b-` is
  trimmed. Rule: once v2b is decided, type `[v2] image_decoupled_from_block` in
  chain.toml (behaviour-identical) so its segment leaves the note.
- **CPU cost.** The bundled 4000-window budget is about twice the current 2000
  windows of CPU scoring per v2 settlement on every validator, so external
  validators should expect longer settlement scoring. The verdict draw is still
  capped by the pool's capacity. `eval_n_windows_after` is load-bounded at
  20000.
- **Testnet** carries `eval_n_windows_after = 300`, its current budget, so the
  vote hash and the explicit verdict draw are rehearsed.

## Second review fixes (2026-11-02)

- Fork guard: `[v2] image_decoupled_feature` pins the vote's feature name
  (Procedure step 2).
- A typed `image_decoupled_from_block` on a config whose rollover is decided on
  chain is checked on the post-rollover era grid
  (`epoch_blocks_after × era_settlements`).
- The lane pool's era-image take is non-destructive: a waiting leg never holds
  a lane of the other image.
- A transport failure during the code-fingerprint probe drains the pod (it is
  not an image mismatch); a funded leg's image refusal never flips it to
  operator billing.
- In-flight restore and the funded receipt check apply the decoupled rule, so
  an old-image leg re-attaches after a post-lock-in restart.
- The bench-only king re-rent runs under its era's contract; MarketCast
  post-switch legs carry `--contract-block`.
- Verified, unchanged: the worker refuses every full-budget leg whose runtime
  image differs from the pin of the contract it resolved (`--contract-block`)
  — `cascade/trainer/worker.py:213` (`is_heat` = `--train-hours` given and the
  role is neither expert nor finetune) and `:301-310`
  (`contract.assert_train_image`). Rolling king and challenger legs pass no
  `--train-hours`, so they are covered; only reduced-budget legs (heats, the
  bench-anneal telemetry leg) skip it; the live trainer always sets `deployed_chain_toml`
  (`cascade/trainer/main.py:321`, `--chain-toml` or `chain.toml`); direct-MoE
  gangs rent their own signed pod profiles (`cascade/direct/compute.py`
  `PodProfile.image`) and are not affected by the `*_after` pins.

## Delta-review fixes (2026-11-02)

- Worker image refusals are structured: exit 2 plus
  `IMAGE_REFUSAL_SENTINEL` as the worker's last stdout line, never free text
  (a generator's exception text cannot trigger lane holds or quarantines).
- Every never-the-wrong-image path is post-switch only; a host or executor
  is blamed only on the trainer's own measured code fingerprint; unverifiable
  probes are "unknown" (wait / retry); the per-leg cap persists and the
  capped leg waits.
- The satellite is a voting process (vote-pin guard, note-budget refusal).
- The note-budget check uses the digit count blocks reach within ~30 days
  (8 digits from ~9,784,000): type v2b's block once decided (or the locked
  rollover) before then.

## Procedure

1. Release this change: validators, trainer and audit together. Nothing changes
   yet.
2. Every node, external validators, the main validator and the trainer,
   deploys the release `chain.toml` values: the typed `[scoring]
   forfeit_from_block = 9166500` / `margin_v2_from_block = 9194400`, `[v2]
   eval_n_windows_after = 4000` and `[v2] image_decoupled_feature =
   "v2b-150ec0d0"`. The main validator's and the trainer's live files are
   edited by hand (the live trainer file lacks `eval_n_windows_after`).
   **Fork guard:** the feature name hashes `eval_n_windows_after`; a file
   without it computes `v2b-6bbd262b` and would signal and tally another
   feature, silently. The loader therefore refuses to start any node whose
   computed name differs from the pinned `image_decoupled_feature`, or whose
   pin is absent while `[activation]` is on and v2 is scheduled.
3. Wait until all six external validators run the release (their notes show the
   `v2b-` segment and no longer show the forfeit or margin-v2 segments).
4. The vote locks in when `threshold` is reached. `status/chain.json`
   `activation.image_decoupled` shows the lock and the switch era start.
5. Either set the `[v2] *_after` pins before the switch (automatic, no restart
   at the switch), or, once the switch era has started, append the current pin
   to `[v2] prior_train_image_digests` and repin by hand (OPERATOR.md).
6. If you need to stop the switch before lock-in: ship a release without the
   segment, or type an override.
