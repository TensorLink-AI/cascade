---
id: DEC-CA-0029
type: decision
title: "The training contract is DECLARED per round, not predicted from local config: validators gate three locked terms and record the rest"
status: active
date: 2026-08-27
tags: [contract, validator, trainer, deploy, audit, consensus]
revisit_when: "a declared term turns out to have a real validator-side consumer after all (then it moves into LOCKED_CONTRACT_FIELDS, which is a coordinated release); or a trainer-declared change is found to move a VERDICT rather than just the checkpoint (the split's core premise fails — re-derive which terms are consensus before shipping anything else); or the trainer stops being a single owner-operated signer (multi-trainer changes the threat model this rests on); or contract-declaration WARNs start being ignored in practice (visibility was the whole price of the trade — if nobody reads the drift, buy the governance back some other way)"
relations: {supersedes_part_of: DEC-CA-0018, relates_to: [DEC-CA-0016, DEC-CA-0019, DEC-CA-0027], depends_on: DEC-CA-0010}
---
Every `[training]` edit moved `contract_digest`, and a validator whose
`chain.toml` disagreed rejected EVERY manifest (`contract_digest_mismatch`).
netuid 91 has six external validators. So re-pinning the worker image — a
routine rebuild for a CUDA or torch patch — meant a coordinated fleet restart,
and the fix waited days on the slowest operator. `scripts/repin_worker_image.sh`
said so in its own deploy protocol.

The gate was not buying what it appeared to. Grep every `cfg.training` read
under `cascade/validator` and `cascade/eval`: there are four, and they resolve
to three terms — `base_arch_digest`, `arch_preset` (+ the size registry), and
`expected_gpu`. `train_image_digest` reaches the validator through NOTHING but
the digest; it is enforced trainer-side by `assert_train_image` against the
pod's injected `CASCADE_TRAIN_IMAGE_DIGEST`, and the one audit consumer
(`audit/rederive.py`) reads it from the auditor's OWN chain.toml. The same is
true of the recipe, the budget, and the corpus mode. The fleet was restarting
in lockstep to distribute facts no validator checked.

DECISION: the trainer PUBLISHES the round's full training contract in the
signed manifest (`contract_body` — exactly the payload `contract_digest`
hashes), and the validator gates on the LOCKED projection only: the three
terms above, plus each extra size's `(arch_preset, expected_gpu)`. Two checks
replace digest equality — the body must re-hash to the digest it declares
(the trainer's two statements must agree), and its locked projection must
equal the local one (read from LOCAL config on our side; a pin sourced from
the declaration would be the trainer gating itself). Everything else is
recorded, signed, and audited, but never a reason to reject a round.

Owner policy, 2026-08-27, and the framing the split serves: **a validator
restart is for an actual scoring change.** `[scoring]`, `[eval]`, and the
cascade envelope decide verdicts and still deploy in lockstep — two validators
disagreeing there FORK. `[training]` decides what the trainer builds, and the
checkpoint is scored identically on the private eval pool no matter which
recipe produced it. `contract_digest` was only ever an ADMISSION gate, never a
scoring input, which is why widening it is monotonically safe for consensus:
divergence comes from a validator REJECTING a round its peers scored, not from
accepting one.

Per-size entries carry only the locked pair, so a `[[training.sizes]]` block's
architecture and budget are declared like any other term ([[DEC-CA-0027]]'s
silicon pin stays locked — it is the matched-hardware gate).

Auditability goes UP, not down. Today `cascade-audit` compares a round against
whatever `chain.toml` says at read time, so a re-pin silently changes the
verdict on an already-published round; the declared body pins a round's terms
to the round itself. The drift is never silent: `check_contract_declaration`
names every non-locked field that differs from the local config, as a WARN —
it cannot FAIL, because these terms are not consensus. That visibility is the
price of the trade, and it is the thing to watch.

What is honestly given up: a validator no longer learns of a `[training]`
change by its config diverging. The residual surface is the owner-operated
trainer quietly moving a declared term — the compute budget being the sharpest
(both arms always share it, so the duel stays fair, but the subnet's spend and
task difficulty move). Considered and rejected as a lock: it is enforceable
only against an operator who already holds the private eval pool, the wallet,
and the signing key, and every such change now lands in every published
receipt and on the dashboard — more visible than a chain.toml commit, not
less. The lock set was put to the owner explicitly (budget and submission
surface offered as additions) and declined: nothing further than the three
terms the validator actually consumes.

Rollout is release-then-activate ([[DEC-CA-0016]]/[[DEC-CA-0019]]'s pattern):
`[scoring] declared_contract_from_block = 0` ships OFF, every round takes the
legacy strict path, and arming it later is the LAST `[training]`-driven change
needing a coordinated deploy. Ship first, let the fleet upgrade at its own
pace, THEN set the block — rounds before it replay strict, rounds after replay
declared, and the audit judges each round under its own rule. A manifest with
no published body always falls back to strict, whatever the block says: the
gate can only relax once the trainer actually declares. Testnet arms at block
1, and that is where the property gets proved — move `train_image_digest` on
the trainer ONLY, leave the validator running, and the round must still score.

`contract_body` rides `canonical_body` under the drop-when-unset convention
(the `bench_scores` / `duel_rank` / eval-pool-pin trick), so a manifest without
it hashes exactly as before and every archived signature stays valid — no
`MANIFEST_VERSION` bump. The golden digest vectors and the signed receipt
fixture are unchanged, which is the proof that no deployed digest moved.

This kills the coordinated-restart half of [[DEC-CA-0018]]'s "trainer+validator
deploy together" — once armed, a WSD or optimizer change is trainer-side. It
does not touch the eval pool, the aggregation, the margin, or the cascade
envelope, all of which still deploy in lockstep.
