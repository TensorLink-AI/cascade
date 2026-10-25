---
id: DEC-CA-0051
type: decision
title: "Port the chain layer to bittensor 11.3.0 with 10.5 bit-compatibility: reveal encoding via bittensor_core (never timelock.encrypt), stake from the typed metagraph (float32, == 10.5 S), weights float32-cast before u16 quantisation"
status: proposed
date: 2026-10-10
tags: [chain, sdk, consensus, operations, miners]
revisit_when: "the whole validator fleet runs 11.x (the float32 cast and the 10.5 weights version key then only matter for reproducibility — keep them unless a deliberate, announced change); or bittensor ships 11.4+/12 (re-run the live differential and the goldens in tests/unit/test_bittensor_v11.py before moving the pin)"
relations: {depends_on: [DEC-CA-0045], relates_to: [DEC-CA-0002, DEC-CA-0036, DEC-CA-0043]}
---

## Decision

`[chain]` pins `bittensor==11.3.0` (11.3.1 is an rc). 11.x is a ground-up
rewrite (one Rust extension, `bittensor-core`; no `bittensor-wallet`,
`async-substrate-interface`, `bittensor-drand`), so `cascade/shared/chain.py`
was re-implemented on the 11.x surface with `ChainClient`'s public interface
and semantics unchanged. Three silent traps are routed around and pinned by
offline goldens (`tests/unit/test_bittensor_v11.py`, fixtures under
`tests/fixtures/bittensor_*`):

1. **Reveal encoding.** The netuid's decoder strips a SCALE compact-length
   prefix from each `RevealedCommitments` entry. `bittensor_core.
   get_encrypted_commitment` seals `compact(len) + UTF-8` (byte-identical to
   10.5's `bittensor_drand`); 11.x's high-level `bittensor.timelock.encrypt()`
   seals RAW UTF-8 — the decoder then eats a real byte and the miner is skipped
   every round while the commit "succeeds". `commit_submission` builds
   `Commitments.set_commitment(netuid, {fields: [{TimelockEncrypted:
   {encrypted, reveal_round}}]})` and submits it **signed by the hotkey**
   (11.x raw calls default to the coldkey, which would file the pointer under
   an unregistered account). Composed call bytes are identical to 10.5's for
   Raw and TimelockEncrypted fields (checked against finney metadata).
2. **Stake.** `neurons.all(lite=True).total_stake` is a different quantity
   (live: 330,736 vs 485,193 for uid 0). The DEC-CA-0045 activation tally
   reads `subnets.metagraph(netuid, block=B).neurons[i].total_stake` — 10.5's
   `metagraph.S` — and rounds it through float32 exactly as 10.5 stored `S`,
   so a mixed fleet sums bit-identical stake.
3. **Weights.** 11.x quantises u16 from float64 (10.5: float32): ~4% of
   vectors moved ±1 LSB on one uid (116/3000 in the study set, 95 of them the
   production decayed-share shape). The vector is cast to float32 before the
   11.x `SetWeights` intent; all 3000 then reproduce 10.5's u16 exactly. The
   implicit 10.5 weights version key (10005000) is passed explicitly.

Failure semantics are kept: 10.5's `set_weights` returned an unsuccessful
response (rate limit, chain rejection) without raising, so 11.x's raised
`bittensor.ChainError` / unsuccessful result is logged and swallowed; only
transport errors surface as cascade's `ChainError` ("weight set failed").
`commit_submission` is the one tightening: a rejected commit now raises
`ChainError` (CLI exit 3) instead of printing "committed".

Signature verification moved to one helper (`cascade/shared/sr25519.py`,
`bittensor.sp_core.Keypair`). A missing/broken SDK RAISES
`SignatureVerifierUnavailable`; it is never a `False` (the intake and the
expert-manifest verifier used to catch the import error and silently reject
every valid signature). sr25519 signatures cross-verify both ways and 10.5
keyfiles load in the 11.x `Wallet` (checked).

## Mapping (10.5 → 11.3)

| ChainClient | 10.5 | 11.3 |
|---|---|---|
| `subtensor()` | `bt.Subtensor(network=…)` + `__del__` defusal | `bt.Subtensor(network)` (GC teardown bounded; `reconnect()` closes the old client off-thread) |
| `current_block` | `get_current_block()` | `.block` |
| `block_hash(B)` | `get_block_hash(B)` | `block_info(B).hash` |
| metagraph reads | `metagraph(netuid, lite=True)` | `subnets.metagraph(netuid, block=, commitments=False)` |
| `validator_stakes(B)` | `meta.S` (float32) | `neurons[i].total_stake.rao / 1e9` → float32 |
| `weights_for_hotkey` | `metagraph(lite=False).W[uid]` | raw `SubtensorModule::Weights[netuid, uid]` normalised in float32 (== `W`) |
| `poll_commitments` | `get_all_revealed_commitments` → raw map fallback | ONE `query_map(RevealedCommitments)` + repo decoder; per-hotkey `query` fallback |
| `read_plain_commitments` | `query_map(CommitmentOf)` | same storage, 11.x `query_map` |
| `pending_commitment` (new) | `.substrate.query(CommitmentOf)` in the miner CLI | `query(CommitmentOf, [netuid, hotkey])` |
| `set_plain_commitment` | `set_commitment` (Raw) | `submit_call(Commitments.set_commitment, Raw<len>, signer="hotkey")` |
| `commit_submission` | `set_reveal_commitment` | `bittensor_core.get_encrypted_commitment` + `submit_call(… TimelockEncrypted, signer="hotkey")` |
| `_set_weights` | `set_weights(...)` | `execute(bt.SetWeights(float32 vector, version_key=10005000))` |
| `subnet_economics` | `subnet()` DynamicInfo | `prices.alpha_price` + `SubnetInfoRuntimeApi.get_dynamic_info` |

## Evidence

Live read-only differential on finney netuid 91 at block 9248428 (HEAD
b43b4fc ChainClient on 10.5.0 vs the ported client on 11.3.0, both pinned to
the block): block hash, block seed, n=256 hotkeys, highest-incentive hotkey,
`poll_commitments` (241 latest / 272 with history), all 256 `validator_stakes`
rows (float repr), the 3 plain readiness notes, every activation tally
(rolling-era-king, forfeit, margin-v2 × dormancy 0/5000), the top-12 permit
holders' weight rows, and `subnet_economics` — all identical.

## Operational

Validators and the trainer upgrade by re-syncing the `[chain]` extra
(`uv sync --all-extras`); no contract, digest or wire-format change, and
10.5 and 11.x validators may run side by side (identical weights, tallies and
reveal reads). Miners must commit with `cascade submit`/`deploy`; reveals
written from 10.5 environments still decode. `requests` (transitive under
10.5) is now declared in `[chain]` and `[deploy]`.
