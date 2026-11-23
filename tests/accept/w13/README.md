# W13 acceptance tests: lane D king publication

The row (plan §3.0, lane D core table), verbatim:

> | W13 | King publication (§11.1): Hippius Hub, R2, HF mirror, signed `champions/lane_d/index.json`, `laned-king-sync.yml`, cards, one-digest check  **(rev5)** A takedown path for the Hub and HF mirrors (licence or rights claim ⇒ unpublish the mirrors, keep the digest and verdict record). | W5e, W3a | ci + testnet | `scripts/laned/accept W13` |

Sources: §11.1 "Where checkpoints live" (the Lane D kings, Lane D index and
Repo cards rows, and "One digest, one identity: vault, public repo, index and
report must agree, and the audit checks it"); §1.2 "Storage" with the
**rev6.4** owner finding (Hippius has no Object Lock, so lane D kings are
**not** locked by a COMPLIANCE copy at crowning; a king's immutability rests
on the content-addressed path, the re-hash on every read, the signed digest
and the R2/HF mirrors); B9 (rev6.4); B13 (the Hub/HF takedown path).

These tests are the CI part of the row. The testnet part (a real crowning
published to the real Hub, R2 and HF) is not here. The builder may touch only
the paths in `PATHS`. No network: the mirrors are in-memory fakes, the vault
is the real `FilesystemStorage` adapter under `tmp_path`, and an autouse
fixture makes any socket connect fail. The test keys come from fixed
`//w13-*` URIs.

## Contract: `cascade/lane_d/publish.py`

Torch-free. It may use the existing lane D intake and storage modules
(`manifest_digest`, `read_manifest`, `entry_prefix`) but doesn't change them.

```python
MIRRORS = ("hub", "r2", "hf")
HF_MIRROR_REPO = "tensorlink-dev/cascade-lane-d-kings"
class PublishError(ValueError):            # .code: a string from the codes below
def hub_repo(ns, context, hotkey) -> str   # f"{ns}/laned-king-c{context}-{hotkey[:8]}"
def publish_king(*, hotkey, digest, vault, mirrors, alert) -> dict
def verify_mirrors(digest, mirrors) -> dict[str, str]
def build_index(kings, signer) -> dict
def verify_index(index, signer_ss58) -> bool
def sync_cards(repo_root, index, signer_ss58) -> list   # the paths written
def check_one_digest(*, index, report, vault, mirrors) -> list[str]
def takedown(index, *, digest, reason, block, mirrors, signer) -> dict
```

### Mirror clients

A mirror is any object with `put(key, data)`, `get(key) -> bytes` (raises
`KeyError` when absent), `list(prefix) -> sorted keys`, `size(key)` and
`delete(key)`. The production adapters (Hippius Hub, R2 public bucket, HF
dataset) implement these five methods. `put` takes exactly a key and bytes,
so the client protocol has no way to request a lock or a retention period.
`mirrors` is a mapping `name -> client` with names from `MIRRORS`.

### `publish_king`: vault to the three mirrors

- It reads the entry from the vault (`lane-d/<hotkey>/<digest>/`) and
  **re-hashes** it: the manifest must hash to `digest`, and every file to its
  manifest sha256. On failure it raises `vault_digest_mismatch`, calls
  `alert({"code": "vault_digest_mismatch", "digest": ..., ...})`, and writes
  nothing to any mirror.
- All three mirrors are required. A mapping without one of them is refused
  with `mirror_missing`, and nothing is written.
- **Content-addressed paths:** every key it writes has `digest` as one
  `/`-separated component. It writes exactly the manifest's files plus
  `manifest.json`, with the vault's bytes.
- **Read-back:** after writing, it re-reads and re-hashes every mirror. A
  mirror holding other bytes is refused with `mirror_digest_mismatch`, and
  `alert` gets a mapping with `code = "mirror_digest_mismatch"`, `mirror =
  <name>` and `digest`. The refusal is a raise, never a logged success.
- On success it returns `{"digest": digest, "mirrors": {"hub": "ok", "r2":
  "ok", "hf": "ok"}}`. Publishing the same king again is a no-op on the
  contents, and two publishes of one king write identical objects. The vault
  is never written.

### `verify_mirrors`: the re-hash on every read

Returns `name -> "ok" | "missing" | "digest_mismatch"` for every given
mirror. It re-reads the bytes each time, with no cached verdict. An
overwritten weights file is `digest_mismatch` on that mirror only. A
consistent rewrite of the weights **and** `manifest.json` is still
`digest_mismatch`, because the manifest no longer hashes to the digest in the
path. With no lock, this check is what makes the path immutable. A mirror
with nothing under the digest is `missing`. A mirror missing one file may be
`missing` or `digest_mismatch`.

### `build_index` / `verify_index`: the signed `champions/lane_d/index.json`

- The index is `{"schema": "lane-d-index/1", "kings": [...], "signer": ss58,
  "signature": hex}`. Each king carries, unchanged, `hotkey`, `context`,
  `digest`, `crown_block`, `lcb`, `gate`, `sentinel` (with its `suite`
  named) and `license` (§11.1). The builder may add fields, for example the
  Hub ref.
- `signer` is a keypair (`.ss58_address`, `.sign(bytes)`), as in
  `lane_d_report.sign_report`. The signature is sr25519 over the canonical
  JSON of the index without `signature` (`sort_keys`, compact separators, no
  NaN), with `signer` inside the signed body.
- `verify_index(index, pinned_ss58)` is `True` only for an untampered index
  signed by that key. A changed LCB, digest, sentinel suite or licence, a
  dropped king, a missing or zeroed signature, another key's signature, or
  another key's index relabelled with the pinned ss58 all give `False`.
  Non-dict input gives `False`, never an exception.
- Refusals: a missing field (including `sentinel.suite`) is
  `index_field_missing`; a digest that isn't 64 lowercase hex is
  `index_field_invalid`; the same digest twice is `index_duplicate_digest`
  (one digest, one identity); a non-finite LCB raises `ValueError`
  (`PublishError` qualifies).
- **Deterministic:** the kings are ordered by `crown_block`, with digest as
  the tie-break, so any input order gives the same canonical body. The input
  isn't mutated. (sr25519 signatures are randomised, so the tests compare
  bodies, not signatures.)

### `sync_cards`: the repo copy (`laned-king-sync.yml`)

- `index is None` means lane D is unarmed and nothing has been published. It
  returns `[]` and the repo tree stays **byte-identical**: no `champions/lane_d/`
  is created (invariant 1, paired with an armed test per invariant 10).
- It first verifies the index under `signer_ss58`. If that fails it raises
  `index_signature_invalid` and writes nothing.
- It writes `champions/lane_d/index.json` (the signed index; it parses back
  equal and still verifies) and, per king,
  `champions/lane_d/<digest>/{PROVENANCE.json, MODEL_CARD.md}`. Nothing else
  is written: no weights in git. `PROVENANCE.json` holds the king's `hotkey`,
  `context`, `digest`, `crown_block`, `lcb`, `license` and `hf_repo` (=
  `HF_MIRROR_REPO`). `MODEL_CARD.md` names the digest and the licence. The
  output is byte-deterministic: two syncs of one index give identical trees,
  and a re-sync changes nothing.

### Nothing claims a lock (rev6.4)

- The index, `PROVENANCE.json` and `MODEL_CARD.md` contain none of the words
  `lock`, `locked`, `object lock`, `retention`, `compliance` or `worm`
  (case-insensitive). The immutability story is content addressing and the
  re-hash, not a lock.
- `cascade/lane_d/publish.py` and `scripts/laned_king_sync.py` contain no S3
  Object Lock API names (`ObjectLockMode`, `ObjectLockRetainUntilDate`,
  `ObjectLockLegalHoldStatus`, `ObjectLockEnabledForBucket`,
  `put_object_retention`, `put_object_legal_hold`,
  `put_object_lock_configuration`, `x-amz-object-lock`,
  `BypassGovernanceRetention`).

### `check_one_digest`: one digest, one identity

Read-only. It makes no mirror call other than `get`/`list`/`size` and
changes neither the vault nor the index. It reads `report["crowned"]` and
`report["boundary"]`; checking the report's signature is the caller's job
(`verify_report`). It returns a list of problem codes, and `[]` means
everything agrees:

| condition | code |
|---|---|
| `report.crowned` is `null` | `report_no_crown` |
| the crowned digest isn't in the index | `index_missing` |
| the index entry's `crown_block` ≠ `report.boundary` | `crown_block_mismatch` |
| the vault entry fails its re-hash | `vault_digest_mismatch` |
| mirror `<m>` is not `ok` | `<m>_missing` / `<m>_digest_mismatch` |

A disagreeing mirror is named and the healthy ones aren't. For a king the
index marks as taken down, an absent Hub or HF copy is not a problem.

### `takedown` (rev5, B13)

`takedown(index, digest=, reason=, block=, mirrors=, signer=)` handles a
licence or rights claim (`reason` `"licence"` or `"rights"`; an empty reason
raises `ValueError`).

- It deletes every Hub and HF object under the digest, so `verify_mirrors`
  then gives `missing` for both. The vault is untouched.
- It returns a **new signed index**. The king's entry keeps every §11.1
  field unchanged and gains `"takedown": {"reason": reason, "block": block}`.
  The input index isn't mutated. The re-synced `PROVENANCE.json` still
  exists and carries the same `takedown`, so the digest and verdict record
  are kept.
- The taken-down king still passes `check_one_digest`, with no `hub_*`,
  `hf_*`, `vault_*`, `index_*`, `report_*` or crown-block codes.
- An unknown digest is refused with `unknown_king`, and no mirror changes.
- What happens to the R2 copy on takedown is the builder's choice; the row
  names the Hub and HF mirrors only.

## Contract: `.github/workflows/laned-king-sync.yml` and `scripts/laned_king_sync.py`

- Triggers: a non-empty `schedule` and `workflow_dispatch`.
  `permissions.contents: write`.
- A step runs `scripts/laned_king_sync.py`, which imports
  `cascade.lane_d.publish`. That's the tested code path, not a copy.
- The only secret it references is `GITHUB_TOKEN`. The sync reads the public
  signed index and writes cards; storage and HF credentials stay with the
  publisher.
- Every `git add` names paths under `champions/lane_d` only, never
  `-A`/`--all`/`-u`.
- Lane A's `.github/workflows/king-sync.yml` doesn't mention lane D
  (invariant 1; this passes on main and must stay green).

## Failure on main (expected)

85 collected, 1 passed, 84 failed, all for a right reason:
`ModuleNotFoundError: cascade.lane_d.publish`, a PATHS module, raised inside
each test body (no PATHS import at module level or in a fixture), or an
`AssertionError` naming the missing workflow, sync script or module file.
The passing test is the lane A king-sync check.
`scripts/laned/accept W13 --expect-fail` → `PASS W13 expect-fail`.
