# Upgrading cascade to Bittensor v11 — assessment

Status: **assessment only** (no decision node yet; adopting this would be
`DEC-CA-0003`). Everything below marked *verified* was checked against
`bittensor==11.0.0` installed from PyPI (2026-07-19), by introspecting the
actual package — not from docs prose.

## TL;DR

The upgrade is **feasible and well-contained**. All chain access already goes
through the `ChainClient` facade in `cascade/shared/chain.py` (plus two
`Keypair` import sites), so the blast radius is one module, two import lines,
tests, and docs. v11 keeps a synchronous client (`bt.SyncClient`) with the
same surface as the async one, so the sync facade survives — no async rewrite
of the validator/trainer/provisioner loops is needed.

Two big wins, one real migration cost:

- **Win 1:** `read("commitments", netuid)` returns per-hotkey rows with a
  decoded `revealed: list[(block, payload)]` from a new Rust decoder — this
  can eventually replace the ~150 lines of tolerant-decode machinery
  (`_raw_revealed_entries`, `_revealed_raw_map`, `_revealed_per_uid`,
  `_decode_reveal_entries`) we built around the v10 payload-length-lottery
  decoder bug. Do not delete the fallbacks until the fix is proven on testnet
  (see Risks).
- **Win 2:** `bt.Policy(allowed_netuids=[...])` hard-bounds what the
  validator/trainer hotkey can do at one choke point, and every write returns
  an `ExtrinsicResult` with a semantic `ErrorCode` (`RATE_LIMITED`,
  `NOT_REGISTERED`, …) instead of string-parsed exceptions. Both are direct
  upgrades to the mainnet-launch hardening list (`docs/MAINNET_LAUNCH.md`).
- **Cost:** `set_reveal_commitment` no longer exists. The miner commit path
  must be rebuilt as `bt.timelock.encrypt(payload, reveal_in=…)` +
  `bt.calls.Commitments.set_commitment(netuid, info)` via `submit_call`, and
  reveal timing moves from *blocks-until-reveal* to *drand rounds / wall-clock
  duration* — a semantic change that touches the round-boundary logic.

Rough effort: **2–4 focused days** — one for the `chain.py` port with unit
tests green, one to two for a testnet-259 soak (commit/reveal roundtrip at
both payload lengths, weight set, poll timing), plus the fleet-coordination
overhead below.

## What v11 actually is (verified)

- One package: SDK + btcli + wallet merged (`bittensor-wallet` and
  `async_substrate_interface` are **gone** as dependencies). The dependency
  tree collapses to `bittensor-core` (Rust extension), `eth-account`, `rich`,
  `typer`, `websockets` — no torch, no numpy, no scalecodec. Installs fine on
  Python 3.11 (cascade requires `>=3.11`).
- API model: **intents** (73 named ops, JSON-schema'd, executed via
  `client.execute_tool(name, args_dict, wallet)`), **reads** (82 named reads
  via `client.read(name, …)`), a **Policy** choke point, generic
  storage/constant accessors (`client.query`, `client.query_map`,
  `bt.storage.*`), and a raw-call escape hatch
  (`bt.calls.<Pallet>.<call>` + `client.submit_call`).
- `bt.SyncClient(network=…, policy=…)` mirrors the async `bt.Subtensor`
  method-for-method (`read`, `query`, `query_map`, `execute_tool`,
  `submit_call`, `block`, `block_info`, `wait_for_block`, `wait_for_epoch`,
  …). **The sync `ChainClient` facade keeps working; only its internals
  change.**
- `Keypair` moved: `from bittensor import Keypair` →
  `from bittensor.wallets import Keypair`. *Verified:* `create_from_uri`,
  `sign`, `verify`, and ss58-only verification (`Keypair(ss58_address=…)`)
  all behave identically — **manifest and receipt signatures remain valid
  across the upgrade; only the import line changes.**
- `Wallet(name=…, hotkey=…, path=…)` — same constructor kwargs as today.
- New chain concept surfaced everywhere: **mechanisms** (`mechid` on
  `set_weights` and the `weights` read; `0` is the default and, for cascade,
  the only one). Pass `mechid=0` explicitly.

## Method-by-method migration map for `ChainClient`

| Facade method | v10 implementation | v11 implementation (verified surface) |
|---|---|---|
| `current_block()` | `sub.get_current_block()` | `client.block()` |
| `block_hash(block)` | `sub.get_block_hash(blk)` | `client.block_info(block).hash` (`BlockInfo` carries `number`, `hash`, `timestamp`, `header`, `extrinsics`) |
| `block_seed(block)` | hash → `seed_from_block_hash` | unchanged (pure function; only the hash source moves) |
| `highest_incentive_hotkey()` | `sub.metagraph(netuid, lite=True)`, scan `meta.incentive` | `client.read("metagraph", netuid=…)` → `Metagraph.neurons` each with `.incentive`, `.hotkey`; or the lighter `read("neurons", netuid, lite=True)` |
| `n_uids()` | `meta.n` | `len(metagraph.hotkeys)` |
| `uid_for_hotkey(hk)` | scan `meta.hotkeys` | `client.read("uid", hotkey_ss58=…, netuid=…)` — one call, no metagraph scan |
| `weights_for_hotkey(hk)` | full metagraph, `meta.W` / `meta.weights` row | `client.read("weights", netuid=…, mechid=0)` → `{validator_uid: {miner_uid: fraction}}`, rows pre-normalised. NOTE: audit compares **support**, not magnitudes — unchanged, but the v10 `W`-vs-`weights` shim dies |
| `poll_commitments()` | `get_all_revealed_commitments` + 3 fallback decode paths | `client.read("commitments", netuid=…)` → per-hotkey rows: `hotkey`, `uid`, `block`, `encrypted`, `reveal_round`, `revealed: list[(block, payload)]`. Keep taking latest reveal per hotkey. Raw fallback stays available: `client.query_map` on `bt.storage.Commitments.RevealedCommitments` |
| `commit_submission(payload, blocks_until_reveal)` | `sub.set_reveal_commitment(…)` | **rebuilt**: `tl = bt.timelock.encrypt(payload, reveal_in=…)` → `info = {"fields": [[{"TimelockEncrypted": {...}}]]}` → `client.submit_call(bt.calls.Commitments.set_commitment(netuid, info), wallet, signer="hotkey")`. Reveal timing is now drand-round/duration-based, **not block-count-based** |
| `_set_weights(weights)` | `sub.set_weights(wallet=…, uids=…, weights=…)` | `client.execute_tool("set_weights", {"netuid": …, "uids": …, "weights": …, "mechid": 0}, wallet)` → `ExtrinsicResult`; branch on `result.error.code` (`ErrorCode.RATE_LIMITED` etc.) instead of wrapping strings |
| `_defuse_substrate_destructor()` | neuters `async_substrate_interface.__del__` hang | package no longer exists → the defusal silently no-ops (it's written version-tolerantly). Delete after cutover; the 2026-07-14 5.5h-freeze failure mode dies with the library |

The version-tolerance shims (`getattr(bt, "subtensor") or bt.Subtensor`,
lowercase `wallet` factory, `W`-vs-`weights`, `_split_commitment`'s
four-shape decode) all become dead weight — v11 is a clean break, so pin
exactly and delete the shims rather than adding a fifth branch to each.

## Files touched

- `cascade/shared/chain.py` — the port. ~Half the module (decode fallbacks)
  is deletable *after* the testnet soak proves the Rust decoder.
- `cascade/shared/manifest.py:417`, `cascade/shared/receipt.py:583` —
  `from bittensor import Keypair` → `from bittensor.wallets import Keypair`.
- `cascade/shared/logging_util.py:44` — re-verify: the "bittensor silences
  other loggers on first import" behaviour and the `bittensor` logger's
  empty-message reconnect spam are v10 artifacts; the restore hook is
  harmless if the behaviour is gone, but check what v11's logging actually
  does before trusting log levels in prod.
- `tests/unit/` — `test_receipt*.py`, `test_audit.py` import `bt.Keypair`
  (one-line fix each). `test_chain.py` / `test_chain_poll.py` fake the v10
  subtensor shape (`sub.substrate.query`, `.value` attrs) — fakes need
  reshaping to the v11 client surface for whatever fallback paths survive.
- `pyproject.toml` — `chain` extra: `bittensor==11.0.0` (keep the exact-pin
  policy and its comment; the reveal-encoding rationale applies with more
  force across a major).
- `uv.lock`, `deploy/Dockerfile`, worker-image digest (pinned in
  `docs/MAINNET_LAUNCH.md` checklist) — rebuild and re-pin.
- `docs/MINER.md`, `docs/VALIDATOR.md` — btcli is now `btcli tx <op>` /
  `btcli query <name>` with `--json` / `--yes` / `--dry-run`; the
  registration and wallet-creation walkthroughs need re-verification against
  the merged CLI (e.g. registration is the `burned_register` op). Also
  MINER.md's warning about "wrong SDK line writes unreadable reveals" needs
  updating to name v11 as the required line post-cutover.

## Risks and open questions

1. **Reveal-timing semantics change.** v10 `blocks_until_reveal=1` is a
   block count; v11 `timelock.encrypt` takes `reveal_in` (duration),
   `reveal_at`, or a drand `reveal_round`. The trainer polls at the epoch
   boundary and filters by `cutoff_block` — confirm on testnet that (a) the
   `revealed` tuples' int is the block the reveal landed (the cutoff filter
   depends on it), and (b) a near-immediate reveal maps cleanly onto the
   round-boundary flow. This is the one place the upgrade touches protocol
   behaviour, not just plumbing.
2. **Cross-version reveal encoding (the reason for the exact pin).** A v10
   miner's reveal must stay readable by a v11 trainer during any transition
   window — and vice versa. The chain store is the same; the decoders differ.
   Test the 2×2 (v10/v11 writer × v10/v11 reader) on testnet 259 with both
   payload lengths (91-char `@hf:` raw rendering and 109-char `@sha256:` hex
   rendering — the exact fixtures in `test_chain.py`) before scheduling the
   fleet cutover.
3. **Don't delete the fallbacks on faith.** The v10 batch decoder poisoned
   the whole netuid on one malformed commitment (field-wide DoS). Verify the
   v11 Rust decoder degrades per-entry, not per-netuid, by writing one
   garbage commitment on testnet. Until then keep `_revealed_raw_map` ported
   onto `client.query_map` as the escape hatch.
4. **Fleet coordination.** This is a dependency bump inside the rsync'd pod
   trees: trainer, validators, provisioner, and the worker image all move
   together (`uv sync --all-extras`, per the operational invariants), and the
   restart must respect the provisioner's pre-boundary trigger window. It
   does **not** change `contract_digest` by itself (that's `[training]`
   config), but the same restart-in-lockstep discipline applies.
5. **Rate limits / weight quantization.** v11 quantizes+normalises weights
   client-side per the `set_weights` schema ("clipped to max-weight limit,
   normalized, quantized"). Cascade's decayed-share vectors are tiny (a few
   nonzero entries), so this should be a no-op — but assert on testnet that
   the on-chain support matches the receipt's recorded vector, since
   `cascade-audit` cross-checks exactly that.
6. **Chain-side compatibility.** v11 targets the current subtensor runtime;
   netuid 91 / testnet 259 run on the live network, so no chain upgrade is
   needed on our side — but `is_fast_blocks`, `mechid`, and epoch reads
   should be sanity-checked against testnet before trusting them in the
   provisioner's timing math.

## Suggested plan

1. **Port (1 day):** bump the pin on a branch, port `chain.py` method-by-
   method per the table (facade signatures unchanged ⇒ zero churn in the
   loops), fix the two Keypair imports, get unit tests green with reshaped
   fakes.
2. **Testnet soak (1–2 days):** on 259 — commit/reveal roundtrip at both
   payload lengths, cross-version 2×2 read/write matrix, one deliberate
   garbage commitment, a full trainer round with weight set, audit
   cross-check of the weight row.
3. **Simplify:** delete the decode fallbacks, version shims, and destructor
   defusal that the soak proves obsolete; add `Policy(allowed_netuids=…)` to
   validator/trainer client construction.
4. **Docs + cutover:** update MINER/VALIDATOR docs to v11 btcli syntax,
   announce the required SDK line to miners, rebuild + re-pin the worker
   image, restart the fleet in lockstep outside the provisioner's trigger
   window.
5. **Decide:** record the go/no-go as `DEC-CA-0003` with
   `revisit_when:` tied to the testnet soak results.

## What v11 offers beyond parity (future, not part of this upgrade)

- `client.wait_for_epoch` / `read("next_epoch_start_block")` /
  `read("blocks_until_next_epoch")` could replace hand-rolled epoch-boundary
  polling in the provisioner and validator loops.
- `plan()` / `--dry-run` previews for the weight-set path would let the
  validator log predicted effects before submitting — useful for the receipt
  trail.
- `read("commitments")` exposes `reveal_round` on sealed payloads — the
  trainer could see *that* a miner has committed (without contents) before
  the reveal lands, enabling earlier round-participation signals.
