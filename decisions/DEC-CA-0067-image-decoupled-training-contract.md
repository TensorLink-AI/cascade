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
("train_image_digest",)` removed. The domain tag means it can never equal a full
digest. `contract_digest` itself is unchanged, so every existing digest is
byte-identical (golden vector).

An audit of `TrainingContractConfig` found only one pure image or runtime pin.
The schedules and `router_slow` are already outside the digest. `expected_gpu`
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

- **Feature.** The name is `v2b-<sha256("image-decoupled/1|train_image_digest|
  eval_n_windows_after=<n>")[:8]>`. It is short because of the note budget below.
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
- **Note byte budget (activation prerequisite).** The plain commitment is one
  128-byte field.
  - The deployed main validator's note today is
    `cascade-ready:1:rolling-era-king:0:0:forfeit-42b1a486:9165600:9166500:margin-v2-d30eb443:9190800:9194400:prelaunch-95-r2:0:0`,
    which is 124 bytes.
  - Adding `:v2b-xxxxxxxx:0:0` (17 bytes) gives 141 bytes. The trimmer drops the
    newest segment, so this vote is NOT signalled.
  - Room has to be freed first. Either type the already-decided forfeit and
    margin-v2 blocks into chain.toml (behaviour-identical, and their segments
    leave the note), or retire the unlocked prelaunch vote. That is an owner
    decision, not part of this change.
  - After lock-in the segment grows to 29 bytes.

## Procedure

1. Release this change: validators, trainer and audit together. Nothing changes
   yet.
2. Free note room (see above).
3. Wait until all six external validators run the release (their notes show the
   `v2b-` segment).
4. The vote locks in when `threshold` is reached. `status/chain.json`
   `activation.image_decoupled` shows the lock and the switch era start.
5. Once the switch era has started, append the current image pin to
   `[v2] prior_train_image_digests`, then repin the worker image
   (OPERATOR.md, "Re-pinning the image inside a v2 seed epoch").
6. If you need to stop the switch before lock-in: ship a release without the
   segment, or type an override.
