---
id: DEC-CA-0036
type: decision
title: "Declared training contract: validators verify a locked projection + self-consistency instead of digest equality — [training] recipe edits become trainer-restart-only"
status: proposed
date: 2026-08-27
tags: [cascade, contract-digest, manifest, validator, deploy, consensus, ops]
revisit_when: "The build lands and the block gate arms (one coordinated release-then-activate — the LAST fleet window the recipe axis needs); or a locked term has to move (arch/size/GPU changes still take the old path); or an audit of a declared-era round fails Tier-2 (the rederive-from-body path is the newest surface)"
relations: {depends_on: DEC-CA-0002, informs: DEC-CA-0035, blocks_nothing: DEC-CA-0033}
---
Status: **plan only. Nothing implemented.** (Owner-reviewed draft 2026-08-27;
supersedes the un-numbered "Declared Training Contract" doc, which collided
with DEC-CA-0029 fork-anneal's ID.)

## Problem

Every `[training]` edit moves `contract_digest`. A validator whose
chain.toml disagrees rejects every manifest (`contract_digest_mismatch`).
With six external validators on netuid 91, a routine worker-image re-pin —
or arming a measured recipe like [[DEC-CA-0035]] — means a coordinated
fleet window and a fix that waits on the slowest operator.

The gate isn't buying what it looks like. Every `cfg.training` read under
`cascade/validator/` and `cascade/eval/` — **five** of them (the original
draft counted four; the fifth is the fix this node adds):

| Read | Site | Gates |
|---|---|---|
| `base_arch_digest` | validator/loop.py:240 | architecture equality |
| `arch_preset` | :557, :893, :1193 | entry → size mapping |
| `size_registry` | :556 | set of legal size names |
| `primary_size` | :565 | fallback for per-size GPU pin |
| `train_seed_salt` | :1304 via `RoundSeeds.derive` (contract.py:149) | the `training_seed` the RECEIPT publishes |

`train_image_digest` reaches a validator through nothing but the digest —
it is enforced trainer-side by `assert_train_image`. Recipe, budget, and
corpus mode likewise have no validator consumer.

**Load-bearing claim (verified against deployed code 2026-08-27):**
`contract_digest` is an admission gate, never a scoring input. A checkpoint
scores identically whatever recipe produced it. Widening admission is
monotonically safe — divergence comes from a validator REJECTING a round
its peers scored, not from accepting one.

## The split

Trainer publishes the round's full contract in the signed manifest
(`contract_body` = exactly the payload `contract_digest` hashes).
Validator runs two checks instead of digest equality:

1. **Self-consistency** — body must re-hash to the digest it declares.
2. **Locked projection** — the body's locked terms must equal ours, read
   from LOCAL config on our side (a pin sourced from the declaration would
   be the trainer gating itself).

| | Terms | Change costs |
|---|---|---|
| **Locked** | `base_arch_digest`, `arch_preset`, `expected_gpu`, each extra size's (preset, expected_gpu), **`train_seed_salt`** | coordinated fleet deploy |
| **Declared** | `train_image_digest`, `lr_schedule`, `base_lr`, `optimizer`, `warmup_fraction`, `batch_size`, `target_train_hours`, `ref_throughput_tokens_per_s`, `max_train_seconds`, `corpus_mode`, `accepted_fields`, `weight_decay`, `ema_decay`, `gen_seed_mix`, `rewarmup_fraction`, the DEC-CA-0035 constants, … | trainer restart only |

`train_seed_salt` is locked (the original draft missed it): the receipt
assembler derives the round's published seeds from the validator's LOCAL
salt — a declared salt change would make every receipt publish a wrong
`training_seed` and break audit replay. Arch-SHAPE fields (`d_model`,
`num_layers`, …) are deliberately NOT in the declared list: they are
folded into the locked `base_arch_digest` and cannot move alone — listing
them as declared would imply otherwise.

**Still lockstep, always:** `[scoring]`, `[eval]`, the cascade envelope.
Those decide verdicts, and disagreement there forks. Don't use this
mechanism to dodge that window.

## Build (seven touchpoints, one release, inert until the block gate is set)

| File | Change | ~Lines |
|---|---|---|
| shared/manifest.py | split `contract_payload()` out of `contract_digest()`; `locked_contract_terms()` + `LOCKED_CONTRACT_FIELDS`; `contract_body` field, canonical_body inclusion, parsing | 95 |
| validator/loop.py | extract `_check_contract()` from `check_manifest`; declared path beside strict; receipt seeds from the declared body's salt (or keep local — salt is locked either way) | 80 |
| audit/checks.py | declared path in `check_contract_digest`; new `check_contract_declaration` | 86 |
| **audit/rederive.py** | **build the Tier-2 replay contract FROM `contract_body`** — the auditor's local config is wrong the first time a declared term moves; this also largely obsoletes `prior_contract_digest`/`contract_from_block` for future transitions (keep them for the archived pre-declared rounds) | ~40 |
| shared/config.py | `[scoring] declared_contract_from_block` + **loader parse + loader round-trip test** (the landed-without-parsing defect hit three PRs in one stack — see OPSLOG 2026-08-27) | 25 |
| trainer/loop.py | pass `contract_body=contract_payload(cfg.training)` | 6 |
| tests/unit/test_declared_contract.py | new | ~280 |

Compatibility: `contract_body` rides `canonical_body` under the
drop-when-unset convention (same trick as bench_scores / duel_rank /
eval-pool pin). A manifest without it hashes exactly as before → archived
signatures stay valid, no MANIFEST_VERSION bump. Golden digest vectors and
the receipt fixture must come through untouched — that is the proof no
deployed digest moved.

Traps (all four from the draft, plus the loader rule made explicit):
1. `contract_payload()` must be idempotent across a JSON round trip, or
   the published body won't re-hash validator-side.
2. JSON has no tuples — `extra_sizes` returns a list. Digest is stable;
   object-identity comparisons aren't. Compare canonical bytes + digest.
3. The block gate compares on the epoch grid (`_epoch_start_block` floors
   `created_block`). Test blocks must survive flooring.
4. The gate key belongs in `[scoring]`. Dropped near `mix_from_block` it
   lands in `[eval]`, loads as 0, and the arming is silently inert — and
   even in the right section it needs the `load_chain_config` parse plus a
   round-trip test, or it no-ops exactly the same way.

## Rollout

One coordinated release-then-activate window — the last one this axis
needs:
1. Ship the code fleet-wide (trainer + all 6 external validators), gate
   unset: behavior byte-identical to today (strict digest equality).
2. Testnet: set `declared_contract_from_block`, run a full round cycle,
   then flip a harmless declared term (e.g. `max_train_seconds` ±1) and
   verify manifests keep landing WITHOUT a validator restart; run
   `cascade-audit` across the boundary.
3. Mainnet: coordinate the block gate with the externals (same playbook as
   DEC-CA-0019's `mix_from_block`), OPSLOG entry.
4. First dividend: the [[DEC-CA-0035]] nine-value recipe cut (and every
   future image re-pin) arms with a trainer restart only.
