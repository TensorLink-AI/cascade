# Auditing cascade rounds — `cascade-audit`

cascade's training is owner-operated in v1 (`OPEN_QUESTIONS.md` #1), so the
trust machinery is built to let **anyone** re-derive the owner's published work
instead of taking it on faith. Every round is a pure function of chain state:

* the **participant set** is the on-chain generator commitments revealed
  strictly before the epoch boundary;
* the **seeds** (`base_seed` → `RoundSeeds`) derive from the epoch-boundary
  block hash;
* the **contract** is the committed `chain.toml` (`contract_digest` /
  `base_arch_digest` / `train_image_digest`);
* the **verdict** is a deterministic paired bootstrap over recorded per-window
  scores;
* the **weights** are `equal_share_vector` of the resulting court.

After each round the validator publishes a signed
[`RoundReceipt`](../cascade/shared/receipt.py) to the manifest bucket under its
own prefix (`receipts/<hotkey>/round-<id>.json` + `receipts/<hotkey>/latest.json`,
plus the shared `receipts/latest.json` pointer — per-validator prefixes mean
concurrent validators never overwrite each other's audit trail) recording all of the
above — including the trainer's manifest verbatim and every per-window score
that fed the KOTH bootstrap. A round the validator *rejected* still gets a
receipt (`"status": "rejected"`) carrying the gate's reason.

`cascade-audit` re-derives a receipt at three tiers:

| tier | cost | needs | verifies |
|------|------|-------|----------|
| 0 (default) | seconds, CPU | repo + `chain.toml` (chain optional) | signatures, seed derivations, contract/arch digests, commitment cutoff, the KOTH verdict recomputed from the receipt's own scores, transition consistency, the weight vector |
| 1 | minutes, CPU | + registry access | + each generator re-run in the sandbox at the round's `generation_seed`; corpus digests byte-compared per (entry, size) |
| 2 | hours, GPU (`[train]` extra) — **experimental** | + a training GPU | + re-train from the contract under the recorded seeds/budget; byte-exact checkpoint comparison on matched `gpu_name` + `train_image_digest`, else eval-score comparison within a documented tolerance |

No credentials are required where the storage allows it: receipts are fetched
with an unsigned S3 request first (falling back to `HIPPIUS_S3_*` if set), or
read from a local file with `--receipt`.

## Quick start

```bash
pip install -e '.[hippius]'        # boto3 for the receipt fetch
cascade-audit latest               # tier 0 on the newest round
cascade-audit round 2269901645662351552 --tier 1
cascade-audit latest --json        # machine-readable; exit 1 on any FAIL
cascade-audit round <id> --validator 5F…   # one validator's receipt specifically
```

Without `--validator`, `latest` reads the shared pointer (the most recently
published receipt, whichever validator wrote it) and `round <id>` discovers the
receipt through the public `receipts/index.json` (legacy un-namespaced rounds
are tried first).

Exit status is nonzero **iff any check FAILs**, so it drops straight into CI:

```yaml
- run: cascade-audit latest --tier 1 --json
```

## Reading the output

Each check reports one of:

* **PASS** — the receipt matches an independent re-derivation.
* **FAIL** — the receipt *contradicts* a re-derivation. This is the signal;
  the exit code turns nonzero.
* **WARN** — could not fully verify (no chain connection, lite node without
  the historical block, pruned commitment history, unpinned signer, a
  stream-mode corpus without `--full-stream`). Deliberately explicit — a check
  never silently passes because its inputs were unavailable.
* **SKIP** — not applicable (e.g. verdict checks on a rejected round).

On a `rejected` receipt, a check that re-detects the recorded rejection reason
(e.g. `manifest-signature` on a `signature_invalid` rejection) reports PASS
with a note that it *confirms* the validator's gate.

## A worked transcript

Tier 0 against a signed round receipt, offline (`--no-chain`), with the two
trust anchors pinned in `chain.toml [manifest]`:

```text
$ cascade-audit round 2269901645662351552 --receipt round-2269901645662351552.json --no-chain
round 2269901645662351552  status=scored  tier=0

  [PASS] status              scored round
  [PASS] receipt-signature   signed by 5CzRdZLnRxWkyWoGZZbfvSCY4mbw1cmsow5q8cgZyGg1aiDq
  [PASS] manifest-signature  signed by 5DpzeqS6r8xy5ajWGJfsPxKEsWEj7bPTN6Bz6knnvez5PPiU
  [PASS] base-seed           base_seed 2269901645662351552 derives from the recorded block hash
  [PASS] round-seeds         generation + training seeds derive from base_seed
  [PASS] epoch-alignment     boundary 21600 on the epoch grid
  [WARN] block-hash-onchain  no chain connection; recorded block hash not verified on-chain
  [PASS] contract-digest     contract_digest d6e93bf5fb306408… matches chain.toml
  [PASS] base-arch-digest    manifest matches the pin; pin recomputes from local source
  [WARN] commit-cutoff       all 2 participant(s) pre-cutoff and entries match the recorded set; chain payloads not cross-checked (no connection)
  [PASS] koth-params         recorded params match chain.toml [scoring]
  [PASS] verdict             lcb=0.40000 margin=0.02000 win=True reproduced over 1 size(s)
  [PASS] transition          transition 'dethroned after 1 consecutive win(s)' consistent with the verdict
  [WARN] weights             vector recomputes; on-chain weights not compared (no chain connection)

  WARN=3  PASS=11
$ echo $?
0
```

Drop `--no-chain` and the three WARNs resolve on-chain: the recorded block
hash is compared against the node, participant payloads against the visible
reveals, and the weight *support* against the validator's current row.

Now tamper with one byte — say the manifest's `contract_digest`:

```text
$ cascade-audit round 2269901645662351552 --receipt tampered.json --no-chain
  ...
  [FAIL] receipt-signature   signature does not verify against 5CzRdZLnRxWkyWoGZZbfvSCY4mbw1cmsow5q8cgZyGg1aiDq
  ...
  [FAIL] contract-digest     manifest contract_digest 0000000000000000… != local chain.toml d6e93bf5fb306408… (different training contract)
  ...
$ echo $?
1
```

Every field of the receipt is inside the signed canonical body, so any
mutation kills the signature *and* trips its specific semantic check (the test
suite asserts one tamper test per check — `tests/unit/test_audit.py`).

## What each tier proves

**Tier 0** proves the round is *internally* honest and matches the public
contract: the verdict really follows from the recorded scores under the
recorded seed and the published `[scoring]` params, the seeds really derive
from the recorded block hash, the participant set respected the deadline, and
the published weight vector is exactly the equal-share function of the
recorded court. What it takes on trust: that the recorded *scores* honestly
describe the published checkpoints (that's Tier 2's job) and, offline, that
the recorded block hash is the real chain's (drop `--no-chain`, or check any
block explorer).

**Tier 1** proves the corpus provenance: the pinned generator, run at the
receipt's `generation_seed` in the same sandbox, reproduces the exact
`corpus_digest` the trainer claimed to have trained on.

* `cache_reuse` corpora byte-compare in one materialisation.
* `stream_cpu` digests cover the *full consumed training budget* (that's what
  makes them binding), so re-deriving one costs about the round's own
  generation time — run with `--full-stream` to spend it; otherwise the check
  WARNs rather than pretending.
* `stream_gpu` is tolerance/same-hardware by design and never byte-compares on
  CPU.

**Tier 2 (experimental)** closes the loop: re-train from the contract under
the recorded seeds and token budget. On a runtime matching both the entry's
recorded `gpu_name` and the contract's `train_image_digest` pin, the
checkpoint must reproduce **byte-identically** (the reference trainer is
deterministic). On other hardware, the retrained checkpoint is scored against
the published one on the receipt's eval slice and must agree within
`TIER2_SCORE_RTOL` (5% relative on geomean(CRPS, MASE) — generous against
kernel-reduction drift, tight against a swapped corpus or checkpoint).

## Trust anchors

Two ss58 addresses in `chain.toml [manifest]` anchor the audit:

* `trainer_hotkey` — must sign the embedded `TrainingManifest`.
* `validator_hotkey` — must sign the `RoundReceipt`. If left empty, the audit
  verifies the signature against the receipt's self-declared signer and WARNs
  that the signer is unpinned (internal consistency only).

Everything else is *derived*, not trusted: digests recompute from the repo,
seeds from the block hash, the verdict from the scores.
