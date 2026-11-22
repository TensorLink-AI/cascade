# W18 acceptance tests: the lane D vault TTL sweeper

W18 (plan §3.0 W18 row, §1.2 "Visibility and retention", §6 D8 and D16,
§17.4 register) builds the job that retires lane D losers:

> vault → owner-only archive (sha256-verified) → delete from the vault; after
> the dispute window, delete from the archive.

Hippius accepts lifecycle rules but doesn't apply them (§1.2), so TTL deletion
is our own job. Automated deletion is Approve-tier unless the owner's standing
approval D16 is decided. These tests are the spec. The builder may touch only
the paths in `PATHS`, which is one new module, `cascade/laned/vault_sweeper.py`.
No config key is added. The vault TTL is the existing `[lane_d] loser_ttl_eras`
(D8, 4), and `[lane_d_ops]` is unchanged (W2b pins its key set).

Some rules for the tests themselves:

- **No network, no buckets.** The in-process tests use two stand-ins:
  - `helpers.FakeS3`, a boto3-shaped, in-memory, versioned multi-bucket S3
    (the w3a fixture S3 model, made writable);
  - the module's own `FsStore` over `tmp_path`.

  The CLI runs as a subprocess with `boto3`, `botocore` and `bittensor`
  unimportable and every socket connect refused.
- **The module is imported only inside test bodies** (`helpers.vs()`).

On main every test except the evidence one fails with a
`ModuleNotFoundError` for `cascade.laned.vault_sweeper`, which is in `PATHS`.
The CLI tests fail on the AssertionError that carries the subprocess's
`ModuleNotFoundError`. `test_evidence_is_a_clean_live_run_on_the_vault` skips
while `LANED_EVIDENCE` is unset.

## Module surface (`cascade.laned.vault_sweeper`)

| name | contract |
|---|---|
| `ARCHIVE_RETENTION_DAYS` | `14` (D8: owner-only archive for the 14-day dispute window) |
| `SweepPolicy(loser_ttl_eras: int, archive_retention_days: int = ARCHIVE_RETENTION_DAYS)` | the policy |
| `policy_from_config(cfg)` | `SweepPolicy` from a loaded `ChainConfig`: `cfg.lane_d.loser_ttl_eras` (overlay-settable, never a constant) |
| `d16_status(plan_path) -> str \| None` | the `Status` cell of the `D16` row in the table headed `\| D \| Status \| Option chosen \| Class \| Decided \|` (plan §17.4), stripped. `None` if the file, the table or the row is missing. The §6 recommendation table is never read as a decision |
| `SweepRefused(Exception)` | has `.code` |
| `ObjectLocked(Exception)` | raised by an adapter's `delete` on an object under unexpired retention |
| `FsStore(root: Path)` | filesystem adapter, see below |
| `S3Store(client, bucket: str, prefix: str)` | boto3-shaped adapter, see below |
| `sweep(*, vault, archive, ledger, policy, now, live, plan_path, log_dir) -> dict` | one run. Returns the action log and writes it to `log_dir/vault-sweep-<run_id>.json` |

### Keys

An adapter's keys are relative to its root or prefix. A vault key
`<hotkey>/<digest>/<file>` (plan §11.1 `lane-d/<hotkey>/<digest>/`, with the
prefix stripped) is archived at the **same relative key** in the archive
store. For example, vault `lane-d/H/D/model.safetensors` goes to archive
`lane-d-archive/H/D/model.safetensors`. `digest` is 64 lowercase hex.

## The retention ledger (`laned_vault_ledger/1`)

The sweeper never infers judgement from storage. It reads a ledger (a snapshot
of lane D state; producing it is a later WP's job):

```json
{"schema": "laned_vault_ledger/1", "era": 20,
 "entries": [{"hotkey": "5F…", "digest": "<64 hex>", "status": "lost", "closed_era": 16}],
 "disputes": ["<64 hex digest>"]}
```

Here `era` is the current era index (int ≥ 0), `closed_era` is the era of
judgement or close (int, or `null` for open statuses), and `disputes` lists
the digests with a pending dispute. A malformed ledger raises
`SweepRefused("ledger_invalid")` before any storage call. Malformed means a
wrong schema, a missing `entries`, a negative era, a bad digest, a bad dispute
digest, or a loser status without an int `closed_era`.

## What is kept, and why (typed reasons)

Each vault object is classified through its entry. Every reason is a keep: the
object stays, gets no archive copy, and is retried on the next run.

| condition | reason |
|---|---|
| any version of the key under unexpired Object Lock retention (COMPLIANCE **or** GOVERNANCE; `retain_until > now`) | `object_lock` (the record carries `retain_until`, ISO 8601) |
| status `submitted` / `ready` / `revealed` / `pending` | `open_entry` |
| status `judging` (in a settlement not yet settled) | `settlement_open` |
| status `king` (kings are published and kept permanently, §11.1) | `king` |
| digest in `disputes` | `dispute_open` |
| loser and `era - closed_era < loser_ttl_eras` | `ttl_not_expired` |
| key not `<hotkey>/<digest>/<file>`, or no ledger entry | `unreferenced` (never deleted: fail closed) |
| any other status | `unknown_status` |
| another file of the same entry can't be swept this run | `entry_incomplete` |

The loser statuses are `lost`, `closed_unjudged`, `ready_expired` (B10) and
`voided`. A loser is **due** when `era - closed_era >= loser_ttl_eras`, so the
boundary is inclusive.

**An entry is swept whole.** If any file of a due entry is locked, or its
archive copy fails, no file of that entry is deleted in that run.

## A due object, live

1. Read the vault object and hash it (sha256).
2. If the archive already holds the key with the same sha256 (read back),
   record `archive`/`already` and write nothing. This is the resume path:
   nothing is archived twice. Otherwise copy it and record `archive`/`done`.
   If the archive copy differs (for example a partial earlier copy), it's
   rewritten.
3. **Verify before any delete.** Read the archive copy back (a GetObject on
   S3) and compare sha256. A failed write gives `archive`/`failed`
   `archive_copy_failed`, and a mismatch gives `archive`/`failed`
   `archive_verify_failed`. Either way there's no delete.
4. Delete from the vault: **every version**, by `VersionId`. The vault has
   Object Lock, so it's versioned, and an unversioned delete only adds a
   delete marker. Never pass `BypassGovernanceRetention`, and never attempt a
   delete on a locked version. A failed delete gives `delete`/`failed`
   `delete_failed`, and the next run resumes at step 2.

Archive records carry `sha256` (64 hex) for the outcomes `done`, `already`
and `planned`.

**Archive expiry.** An archive object written by the sweeper more than
`archive_retention_days` before `now` is deleted (all versions). The record
is `store: "archive"`, `delete`/`done`, counted as `expired`. Otherwise it's
kept with `archive_retention_not_expired`, or with `dispute_open` if its
digest is disputed. The archive time is the sweeper's own record of the write.
A re-run that finds a verified copy (`already`) does not reset it. An archive
object whose archive time the sweeper can't establish is kept with
`archive_age_unknown`.

## Dry-run, live, D16

- `live=False` (dry-run): the same classification, with `archive`/`planned`
  and `delete`/`planned` in place of the writes. It makes **no** storage
  write, delete or abort, and on fs it doesn't touch either tree, including
  creating the archive root. D16 isn't needed. `d16` in the log is
  `d16_status(plan_path)` when a plan is given, else `null`.
- `live=True`: if `d16_status(plan_path) != "decided"`, or `plan_path` is
  `None` or missing, it raises `SweepRefused("d16_not_decided")` **before
  any storage call**. The log's `d16` is `"decided"`.
- An archive store that overlaps the vault raises
  `SweepRefused("archive_overlaps_vault")` with no write. That's the same
  bucket with nested or equal prefixes, or an fs root equal to, inside or
  containing the vault root.

## Owner-only archive

- **S3:** every archive write (`put_object`, `copy_object`,
  `create_multipart_upload`) carries `ACL="private"` and no `Grant*`
  argument. The sweeper never writes to the vault. The allowed client surface
  is `list_objects_v2`, `list_object_versions`, `head_object`, `get_object`,
  `get_object_retention`, `put_object`, `copy_object`, the multipart calls
  (`create_multipart_upload`, `upload_part`, `upload_part_copy`,
  `complete_multipart_upload`, `abort_multipart_upload`), `delete_object`,
  `delete_objects` and `get_paginator`. Anything else, such as ACL, policy or
  public-access calls, fails the test. Listings are paginated (the fake
  returns 2 items per page). A client error is recognised by its `.response`
  (the botocore `ClientError` shape), not by its class.
- **fs:** the archive root and every directory under it are mode `0700`, and
  every file is `0600`, even under umask `022`.

## `FsStore(root)`

An object is the regular file `root/<key>`. The adapter may keep its own
metadata (lock state, archive time) only in a directory `root/.laned-meta/`,
which is never listed as objects.

- `put(key, data: bytes, *, lock_mode=None, retain_until=None)` writes an
  object, optionally under retention. Tests use it to seed a locked vault
  object.
- `delete(key, *, now)` raises `ObjectLocked` while `retain_until > now`.

Plain files seeded directly into `root` are unlocked objects.

## The action log (`laned_vault_sweep/1`, strict JSON, one file per run)

The top-level keys are exactly `schema`, `run_id` (16 lowercase hex, unique
per run), `mode` (`dry-run` | `live`), `now`, `started_at` and `finished_at`
(ISO 8601 with timezone), `d16`, `policy`, `ledger_era`, `vault`, `archive`,
`actions` and `counts`.

- `policy` is `{"loser_ttl_eras", "archive_retention_days"}`.
- `vault` and `archive` are `{"kind": "s3", "bucket", "prefix"}` or
  `{"kind": "fs", "root": str(root)}`.
- `actions` is a list of records with at least `store` (`vault` |
  `archive`), `key` (relative), `op` (`archive` | `delete` | `keep`),
  `outcome` and `reason`.
  - Outcomes: archive `done|planned|already|failed`, delete
    `done|planned|failed`, keep `kept`.
  - `reason` is `null` or a typed reason: a keep reason above, or one of
    `archive_copy_failed`, `archive_verify_failed`, `delete_failed`,
    `object_lock`.
  - Every vault object listed at the start of the run gets at least one
    record.
- `counts` holds `archived` (archive done), `deleted` (vault delete done),
  `expired` (archive delete done), `kept`, `failed` and `planned`, all
  re-derived from `actions`.

The log text never carries a URL, signature or credential, and it has no
NaN or Infinity.

## CLI

```
python -P -m cascade.laned.vault_sweeper --vault-dir D --archive-dir D --ledger F \
    --config chain.toml --log-dir D (--dry-run | --live) [--plan P] [--now ISO] [--evidence-out F]
```

- **Arguments.**
  - Exactly one of `--dry-run` / `--live` must be given; otherwise it exits 2.
  - `--config` is loaded with `load_chain_config`, then `policy_from_config`.
  - `--plan` defaults to `docs/plans/LANE_D_PLAN.md`.
  - `--now` defaults to the current UTC time.
- **Output and exit codes.**
  - stdout is the action log, and the exit code is 0, or 1 if any outcome is
    `failed`.
  - A refusal prints `{"refused": "<code>"}` and exits 2.
- **`--evidence-out F`** (live only) writes
  `{"schema": "laned_w18_evidence/1", "wp": "W18", "passed": <no failed outcome>, "log": <the log>}`.
- **The fs mode needs no network and doesn't import boto3.**
- **The live S3 mode** is the one the owner runs. Its bucket and prefix come
  from `[lane_d_ops] vault_bucket`/`vault_prefix` and the owner-only archive
  location. Credentials are env-only, never argv, never logged.

## Evidence (L4: W18 runs in "ci (fs adapter) + owner")

`scripts/laned/accept W18 --evidence evidence/w18.json`, after the owner's
first live run against the real vault. `helpers.evidence_problems`
re-derives the evidence from the embedded log:

- the log shape is valid;
- `mode == "live"` and `d16 == "decided"`;
- the vault is S3 bucket `cascade-lane-d-vault`;
- the archive is S3 and doesn't overlap the vault;
- every vault delete follows a verified archive record (`done`/`already`)
  for that key, and no locked key is deleted;
- `passed` is true exactly when no outcome failed.

An fs-adapter run's evidence is rejected. **D16** is decided on main
(§17.4: "Standing approval for automated TTL deletion granted", owner,
2026-11-21), which is what makes a live run legal.
