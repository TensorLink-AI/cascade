# W3a acceptance tests: the live Hippius storage probe (owner WP)

W3a answers the plan §1.2 storage questions against **live Hippius S3**:
presigned UploadPart, the single-PUT size limit, Object Lock on multipart, a
prefix-scoped bucket policy, sub-token scope, `UploadPartCopy` of a ~2.5 GB
object (locking a king by copy at crowning) and whether COMPLIANCE retention
can be shortened. The answer is the decision record
`decisions/STORAGE-LANE-D.md`, which names **exactly one** upload path.

It runs in `owner (live Hippius)`: the owner runs the probe, CI only verifies
its evidence. These tests are the spec. The builder may touch only `PATHS`:
`cascade/laned/storage_probe.py` and `decisions/STORAGE-LANE-D.md`.

The `test_evidence_*` and `test_decision_*_with_evidence` tests **skip**
(with a clear reason) when `$LANED_EVIDENCE` is unset, so the probe and the
decision template can merge and `accept W3a` goes green (L3) before the
owner's run. L4 needs a green `accept W3a --evidence evidence/w3a_probe.json`
run on main (W0c). When `$LANED_EVIDENCE` **is** set, a missing file, invalid
JSON, `source: "fixture"`, a hand-edited verdict, evidence without a
`verified_live` record from `storage_probe verify` (or edited after it), an
unfilled decision record
or a decision that names a path the evidence does not support fails hard.
Every other test runs offline: the probe is driven only in `--dry-run` mode,
with `boto3`, `botocore` and `bittensor` made unimportable and every socket
connect refused (a `sitecustomize` the tests put first on `PYTHONPATH`).

## The probe

```
python -P -m cascade.laned.storage_probe --bucket <throwaway lock-enabled bucket> \
    --prefix <throwaway prefix>/ --other-bucket <a bucket outside the sub-token's scope> \
    --out evidence/w3a_probe.json --confirm-live \
    [--endpoint https://s3.hippius.com] [--region decentralized] \
    [--lock-retention-seconds N] [--big-object-bytes N] [--part-bytes N] \
    [--single-put-bytes N,N,...] [--dry-run [--fixture MODEL]]
```

### Safety rules (each one is tested)

- **Credentials come from the environment only, by name:** the main key is
  `HIPPIUS_S3_ACCESS_KEY` / `HIPPIUS_S3_SECRET_KEY` (the names
  `cascade.shared.hippius.S3Config` uses), the sub-token is
  `HIPPIUS_PROBE_SUBTOKEN_ACCESS_KEY` / `HIPPIUS_PROBE_SUBTOKEN_SECRET_KEY`
  (a read-write sub-token the owner creates in the Hippius console, scoped to
  `--bucket`). Their values, presigned URLs, signatures and `Authorization`
  headers are never printed, logged or written to the evidence. The evidence
  records the endpoint **host** only. No `.env` / `dotenv` reading.
- **No bucket administration.** The probe never calls `create_bucket`,
  `delete_bucket`, `put_bucket_versioning`, `put_object_lock_configuration`,
  `put_bucket_acl`, `put_object_acl`, `put_public_access_block`,
  `delete_public_access_block` or `put_bucket_lifecycle_configuration`, and
  never uses `S3Store` / `_ensure_bucket` (which create missing buckets). The
  owner creates the throwaway bucket, **with Object Lock enabled**, first.
- **A throwaway bucket only.** `--bucket` must not be any bucket named in the
  repo's `chain.toml` (every string value of a key ending in `bucket`, any
  table, e.g. `cascade-manifests`, `cascade-lane-d-vault`): exit 2.
  `--other-bucket` (only ever listed with the sub-token, never written) must
  differ from `--bucket`: exit 2. A fixture whose `schema` is not
  `laned_w3a_fixture/1` is exit 2.
- **A throwaway prefix only.** `--prefix` matches
  `^[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)*/$`, does not start with
  `lane-d/` (the real vault layout), and must hold **no objects** when the
  run starts (the probe lists it first; a non-empty prefix is exit 2 and
  nothing is written). Every object the run creates is under
  `<prefix><run_id>/`, where `run_id` is 16 random lowercase hex.
- **Bucket policy:** the policy probe refuses to touch a bucket that already
  has a bucket policy (result `error`, no `PutBucketPolicy` request). It puts
  a probe policy and always ends with `DeleteBucketPolicy`
  (`policy_removed: true`).
- **No bucket default retention (RV4).** Before its first write (after the
  empty-prefix check), the probe reads the bucket's Object Lock
  configuration once (GetObjectLockConfiguration). If
  `ObjectLockConfiguration.Rule.DefaultRetention` is present, in **any**
  mode (`COMPLIANCE` or `GOVERNANCE`) and for any period (`Days` or
  `Years`), every object the run writes would inherit that lock (~11 GB of
  payloads undeletable for the period), so the run refuses: exit 2, the
  typed code `bucket_default_retention` on stderr, no evidence, **no
  object, upload or policy written**. A `Rule` without `DefaultRetention`,
  or no lock configuration at all (`ObjectLockConfigurationNotFoundError`,
  404: `lock_enabled: false`), proceeds. Any other answer (403, 5xx, a
  transport error) is exit 2 with code `bucket_lock_unreadable` (fail
  closed: it can't tell). The read is recorded in the evidence as
  `bucket_lock` (below). The in-process surface is
  `storage_probe.bucket_lock_preflight(client, bucket) -> {"lock_enabled",
  "default_retention": null, "http_status", "request_id"}` on a
  boto3-shaped S3 client; it makes exactly one call,
  `get_object_lock_configuration(Bucket=bucket)`, and raises
  `storage_probe.UsageError` whose `code` attribute (and message) is
  `bucket_default_retention` / `bucket_lock_unreadable`. With
  `--fixture`, the model's `default_retention` stands in for the rule. So
  the probe can never lock more than its two ≤ 4096-byte objects, for ≤ 1
  day.
- **COMPLIANCE locks:** only the two lock probes create locked objects, each
  **≤ 4096 bytes**, with the shortest retention the run allows:
  `--lock-retention-seconds` defaults to `86400` (one day) and must lie in
  `[60, 86400]` (exit 2 otherwise), so a typo can't lock anything for longer
  than a day. They cannot be deleted before their retain-until date; the run
  reports each one (`locked_objects`).
- **Cleanup:** at the end the probe deletes every object version and delete
  marker under `<prefix><run_id>/` and aborts every unfinished multipart
  upload there. Only COMPLIANCE-locked objects may remain
  (`cleanup.remaining_unlocked == []`).
- **Live mode** needs `--out` and `--confirm-live` (it creates real objects,
  some locked), and both main credential variables; otherwise exit 2 before
  any network use or `boto3` import. `--endpoint` with userinfo, a query or a
  fragment is exit 2.

### `--dry-run`

Opens no connection and imports no `boto3`.

- **Without `--fixture`:** prints the plan (`would_do`: one list of
  operation descriptions per probe name) and writes nothing.
- **With `--fixture MODEL`:** runs every probe, its cleanup and the evidence
  builder against an in-process backend that behaves as the model file says
  (shape: `fixtures/hippius_model.json`, below), and writes the evidence it
  would write, with `source: "fixture"` and `passed: false`, which the
  verifier never accepts. Request ids from this backend start with
  `fixture-`. Fixture payloads are small: the fixture-mode defaults are
  `--big-object-bytes 1048576`, `--part-bytes 262144`,
  `--single-put-bytes 524288,1048576`, and any size above 16 MiB is exit 2,
  so a fixture run stays light on the prod box. (The presigned UploadPart
  probe always uses parts of 5 MiB and 1024 bytes: 5 MiB is the S3 minimum
  non-final part.)

Live defaults: `--big-object-bytes 2684354560` (2.5 GiB, the BF16 entry
size), `--part-bytes 268435456` (256 MiB; Hippius caps a part at 512 MiB and
an upload at 10,000 parts), `--single-put-bytes 536870912,2684354560`.
Payloads are a deterministic byte pattern streamed from memory in chunks
(never a 2.5 GB buffer); sha256 is computed while streaming.

### The fixture model `laned_w3a_fixture/1`

```json
{"schema": "laned_w3a_fixture/1", "endpoint_host": "s3.hippius.com",
 "region": "decentralized", "prefix_object_count": 0,
 "bucket_lock_enabled": true,
 "default_retention": null | {"mode": "COMPLIANCE | GOVERNANCE", "days": N} | {"mode": "...", "years": N},
 "bucket_has_policy": false,
 "presigned_upload_part": "supported | rejected",
 "single_put_max_bytes": 5368709120, "copy_object_max_bytes": 5368709120,
 "upload_part_copy": "supported | rejected",
 "lock_on_multipart": "applied | ignored | rejected",
 "bucket_policy": "enforced | accepted_not_enforced | rejected",
 "subtoken": "bucket_scoped | unscoped | null",
 "compliance": "unshortenable | shortenable"}
```

Meanings: `rejected` presigned parts answer 403; a single PUT (or a single
CopyObject) larger than its max answers 400 `EntityTooLarge`; a rejected
UploadPartCopy answers 501 `NotImplemented`; `ignored` lock headers create an
unlocked object, `rejected` ones make CreateMultipartUpload answer 400; an
`accepted_not_enforced` policy is stored (204) but the denied write still
answers 200, a `rejected` one answers 501; `unscoped` lets the sub-token list
`--other-bucket` (200), `null` means no sub-token is configured;
`shortenable` lets PutObjectRetention move retain-until earlier.
`default_retention` is optional (absent = `null`): a non-null value is the
bucket's default retention rule (exactly one of `days` / `years`, a positive
int; any other shape is exit 2), and the run refuses it
(`bucket_default_retention`) before any write.

### Stdout and exit codes

Stdout is one JSON object: `dry_run`, `fixture`, `out`, `endpoint_host`,
`bucket`, `prefix`, `run_id`, `would_do`, `results` (probe name → result),
`complete`, `recommended_path`, `locked_objects`, `cleanup`, `passed`.
Exit 0 = `complete` (every probe recorded `pass` or `fail`, cleanup left no
unlocked object, no probe policy left on the bucket); 1 = a probe ended in
`error` or cleanup is incomplete; 2 = usage, refusal, input or credential
error.

## Evidence `laned_w3a_probe/1`

```json
{"schema": "laned_w3a_probe/1", "wp": "W3a", "source": "live",
 "endpoint_host": "s3.hippius.com", "region": "decentralized",
 "bucket": "<throwaway>", "prefix": "<prefix>/", "run_id": "<16 hex>",
 "started_at": "<ISO 8601 with offset>", "finished_at": "<ISO 8601 with offset>",
 "tool": {"module": "cascade.laned.storage_probe", "git_sha": "<40 hex>", "git_dirty": false},
 "params": {"big_object_bytes": 2684354560, "part_bytes": 268435456,
            "single_put_bytes": [536870912, 2684354560],
            "lock_retention_seconds": 86400, "presign_expires_seconds": 3600},
 "bucket_lock": {"lock_enabled": true, "default_retention": null,
                 "http_status": 200, "request_id": "<id>"},
 "probes": [{"name": "<probe>", "result": "pass | fail | error", "passed": true,
             "reason": "<text>",
             "requests": [{"op": "UploadPart(presigned)", "http_status": 200, "request_id": "<id>"}],
             "objects": [{"key": "<prefix><run_id>/...", "size": 5243904, "sha256": "<64 hex> | null"}],
             "observations": {"...": "per probe, below"}}],
 "locked_objects": [{"key": "...", "version_id": "...", "mode": "COMPLIANCE",
                     "retain_until": "<ISO 8601>", "size": 1024,
                     "etag": "<HEAD ETag, as returned>", "sha256": "<64 hex>"}],
 "cleanup": {"deleted_versions": 9, "aborted_uploads": 0,
             "remaining_unlocked": [], "remaining_locked": ["<key>", "<key>"]},
 "recommended_path": "presigned_multipart | single_presigned_put | upload_relay",
 "passed": true,
 "verified_live": {"...": "written by `storage_probe verify`, below"}}
```

`bucket_lock` is the pre-write GetObjectLockConfiguration read
(`default_retention` is always `null` in evidence: a bucket with a rule is
refused before evidence exists; `lock_enabled` agrees with the lock probes'
`bucket_lock_enabled`). Each `locked_objects` entry names its locked
`version_id`, the HEAD `ETag` of that version and the sha256 of its bytes
(equal to the lock probe's `objects[].sha256` for that key): `verify`
compares all of them with the bucket. `verified_live` is absent from the
probe's own output; `verify` adds it.

`probes` has exactly one entry for each of: `presigned_upload_part`,
`single_put_limit`, `object_lock_multipart`, `prefix_bucket_policy`,
`subtoken_scope`, `upload_part_copy_2_5gb`,
`compliance_retention_unshortenable`. `passed` (per probe) is
`result == "pass"`. A request id is the response's `x-amz-request-id`,
falling back to `x-amz-id-2`, then `x-request-id`; every request with an
`http_status` carries one.

### Per-probe observations and the verdict rule

The verifier re-derives every `pass`/`fail` from the observations (a
recorded `error` stands, and makes the run incomplete). Statuses are ints or
`null` (request not made).

| probe | observations | `error` when | `pass` when (else `fail`) |
|---|---|---|---|
| `presigned_upload_part` | `create_status`, `url_expires_seconds`, `authorization_header_sent` (must be false), `parts` [{`part_number`, `size`, `http_status`, `etag_returned`}], `complete_status`, `uploaded_sha256`, `object_size`, `object_sha256` | `create_status != 200` (credentialed CreateMultipartUpload failed) | ≥ 2 parts numbered 1..n, each 200 with an ETag, complete 200, `object_size` = sum of part sizes, `object_sha256 == uploaded_sha256` |
| `single_put_limit` | `required_bytes` (= `params.big_object_bytes`), `authorization_header_sent` (false), `attempts` [{`size`, `http_status`, `error_code`, `object_size`, `sha256`}] (presigned PUTs, sizes a prefix of `params.single_put_bytes`, ascending, none after the first rejection), `max_accepted_bytes`, `min_rejected_bytes` | never | `max_accepted_bytes ≥ required_bytes` (accepted = 200 and `object_size == size`) |
| `object_lock_multipart` | `bucket_lock_enabled`, `requested_mode` (`COMPLIANCE`), `requested_retain_until`, `create_status`, `complete_status`, `head_mode`, `head_retain_until`, `object_size` (≤ 4096) | the bucket has no Object Lock | create 200, complete 200, `head_mode == "COMPLIANCE"`, `head_retain_until` = requested (±1 s) |
| `prefix_bucket_policy` | `preexisting_policy`, `policy` (the document put: a `Deny` whose `Resource` is under `arn:aws:s3:::<bucket>/<prefix><run_id>/`), `credential` (`subtoken`/`main`), `put_policy_status`, `denied_put_status`, `allowed_put_status`, `policy_removed` (must be true) | a bucket policy already existed | put 200/204, denied write 403, allowed write 200 |
| `subtoken_scope` | `subtoken_present`, `other_bucket` (≠ `bucket`), `other_bucket_exists` (main key HeadBucket 200), `in_bucket_put_status`, `in_bucket_get_status`, `other_bucket_list_status` | no sub-token, or `other_bucket` does not exist for the main key | in-bucket put 200 and get 200, other-bucket list 403 |
| `upload_part_copy_2_5gb` | `source_status`, `source_size` (= `params.big_object_bytes`), `source_sha256`, `copy_object_status`, `copy_object_error_code` (single CopyObject, informational), `parts` [{`part_number`, `range_start`, `range_end`, `http_status`}], `complete_status`, `dest_size`, `dest_sha256` | the source upload failed (`source_status != 200`) | parts numbered 1..n with contiguous ranges covering `[0, source_size)`, none longer than `params.part_bytes`, each 200; complete 200; `dest_size == source_size`; `dest_sha256 == source_sha256` (a streamed GET) |
| `compliance_retention_unshortenable` | `bucket_lock_enabled`, `put_status`, `mode`, `retain_until`, `shorten_to` (< `retain_until`), `shorten_status`, `shorten_bypass_status` (with `BypassGovernanceRetention`), `delete_version_status`, `retain_until_after`, `object_size` (≤ 4096) | no Object Lock, or the locked PUT failed / did not lock | shorten, bypass-shorten and versioned delete all ≥ 400, and `retain_until_after == retain_until` (±1 s) |

### The rest of the verifier (`helpers.evidence_problems`)

- the keys above; `schema`, `wp: "W3a"`, `source: "live"`; `endpoint_host`
  a bare lowercase host name; `bucket` not a `chain.toml` bucket; `prefix`
  and `run_id` as above; timestamps ISO 8601 with an offset, started ≤
  finished;
- `tool.git_sha` is a commit in this repository that contains
  `cascade/laned/storage_probe.py`, and `git_dirty` is false;
- live sizes: `big_object_bytes ≥ 2_500_000_000`, `5 MiB ≤ part_bytes ≤ 512
  MiB`, at most 10,000 parts; `lock_retention_seconds` in `[60, 86400]`;
  `presign_expires_seconds` in `[1, 604800]` (Hippius: 7 days max);
- every `pass`/`fail` probe has requests (each with a status carries a
  request id, never `fixture-…` in live evidence) and objects (keys under
  `<prefix><run_id>/`, sha256 64 hex or null);
- `locked_objects`: mode `COMPLIANCE`, size ≤ 4096, retain-until after
  `started_at` and no later than `finished_at + lock_retention_seconds +
  60 s`, keys belonging to the two lock probes, and every object those
  probes observed as COMPLIANCE-locked listed; `cleanup.remaining_locked` ⊆
  the locked keys;
- `recommended_path` is recomputed: `presigned_multipart` if
  `presigned_upload_part` passed, else `single_presigned_put` if
  `single_put_limit` passed, else `upload_relay`;
- `passed` is recomputed: every probe `pass`/`fail`, `remaining_unlocked ==
  []`, `source == "live"`;
- no string anywhere in the file contains a URL, `X-Amz-Signature`,
  `X-Amz-Credential`, `X-Amz-Security-Token`, `AWSAccessKeyId`,
  `Signature=` or `Authorization:`, or the value of any `HIPPIUS_*`
  variable set in the verifier's environment;
- `bucket_lock` as above (all four keys, a status and request id, never
  `fixture-` in live evidence, `default_retention: null`);
- `locked_objects[]` carry `version_id`, `etag` and `sha256` as above;
- **the `verified_live` record** (next section) is present and checks out.
  (`evidence_problems(..., require_verified=False)` checks the body only;
  the offline tampering tests use it. Acceptance never does.)

A `--dry-run --fixture` run relabelled as live (`source`, `git_dirty`,
request ids, `passed`: `helpers.relabel_as_live`) has a self-consistent
body, and is **refused**: it has no `verified_live` record, and `verify`
refuses it against any bucket that does not hold its locked objects.

## `storage_probe verify`: the check against the live resource (plan §17.1 L4)

```
python -P -m cascade.laned.storage_probe verify --evidence evidence/w3a_probe.json \
    [--endpoint https://s3.hippius.com] [--region decentralized]
```

The owner runs it with `HIPPIUS_S3_ACCESS_KEY` / `HIPPIUS_S3_SECRET_KEY`
set, after the probe and **before the locks expire** (within
`lock_retention_seconds` of the run, one day by default). It is read-only
(no put, copy, delete, retention or policy call) and refuses before any
network use or `boto3` import (exit 2) when the main credentials are
missing, when `--evidence` is missing or not JSON (the message names the
file / says "evidence"), or when the evidence `source` is not `live`
(code `not_live`). Otherwise it calls `verify_live` with a boto3 client
for the evidence's endpoint. On success it rewrites `--evidence` in place
with `verified_live` added (replacing an older one) and exits 0; on a
refusal it prints the code on stderr, leaves the file untouched and exits
1.

In-process: `storage_probe.verify_live(evidence: dict, client, *, now:
datetime | None = None) -> dict` returns the `verified_live` record (it
does not mutate `evidence`), or raises `storage_probe.VerifyRefused`, whose
`code` is one of:

| code | when |
|---|---|
| `not_live` | `source != "live"`; raised before any client call |
| `no_locked_objects` | `locked_objects` is empty (nothing in the bucket vouches for the run) |
| `locked_object_missing` | a recorded `(key, version_id)` can't be read from the evidence's `bucket` (HEAD, GetObjectRetention or GET answers an error, any exception) |
| `locked_object_mismatch` | the bucket's version differs from the evidence: HEAD `ETag`, `ContentLength` (`size`), `ObjectLockMode` / GetObjectRetention `Mode` (must be `COMPLIANCE`), `ObjectLockRetainUntilDate` / `RetainUntilDate` (±1 s of `retain_until`), or the sha256 of the GET body |
| `lock_expired` | `now` (default: the current UTC time) is not before a locked object's `retain_until` |
| `run_root_not_clean` | listing every version under `<prefix><run_id>/` (`list_object_versions`, `Prefix=` the run root) finds a key that is not a locked object |

For each locked object it calls `head_object`, `get_object_retention` and
`get_object`, each with `Bucket=<evidence bucket>, Key=<key>,
VersionId=<version_id>`. Request ids come from `ResponseMetadata.RequestId`,
falling back to the `x-amz-request-id` / `x-amz-id-2` / `x-request-id`
response header.

The record:

```json
{"schema": "laned_w3a_verify/1", "verified": true,
 "body_sha256": "<sha256 of the canonical evidence body>",
 "verified_at": "<ISO 8601 with offset: now>",
 "tool": {"module": "cascade.laned.storage_probe", "git_sha": "<40 hex>", "git_dirty": false},
 "endpoint_host": "<as the evidence>", "bucket": "<as the evidence>", "run_id": "<as the evidence>",
 "objects": [{"key": "...", "version_id": "...", "mode": "COMPLIANCE",
              "retain_until": "<ISO 8601, from the bucket>", "size": 1024,
              "etag": "<from HEAD>", "sha256": "<of the GET body>",
              "request_ids": ["<HEAD id>", "<GetObjectRetention id>", "<GET id>"]}],
 "listing": {"prefix": "<prefix><run_id>/", "keys": ["<every key listed there>"],
             "request_id": "<id>"}}
```

**Binding: hash-bound, not signed.** `body_sha256` is the sha256 of the
evidence object *without* its `verified_live` key, serialised as
`json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`
and encoded UTF-8 (`helpers.canonical_body`); it doesn't depend on the
file's whitespace or key order. Why a hash, not a signature: the only key
that could sign is a wallet hotkey, and the probe must stay
bittensor-free and never read a keyfile (a pinned test, and the repo's
key rules); and whoever runs `verify` would hold the signing key anyway,
so a signature would add no protection against the person running it. What
the binding has to stop is an edit after verification: any change to the
body (a verdict, the recommended path, a request id) breaks `body_sha256`.
Relabelling is stopped by the bucket: only a bucket that really holds the
COMPLIANCE-locked versions with those ETags and bytes yields a record, and
COMPLIANCE locks can't be removed early. A hand-written record is not
detectable offline; the owner runs `verify` and opens the evidence PR.

The offline check (`helpers.verified_live_problems`) requires every key
above; `schema`, `verified: true`; `body_sha256` equal to the recomputed
one; `endpoint_host`, `bucket` and `run_id` equal to the evidence's; a clean
`tool` (40-hex sha, a commit here containing the module); `verified_at`
not before `finished_at` and before every locked object's `retain_until`;
`objects` exactly the locked objects, each with the evidence's
`version_id`, `etag`, `sha256`, `size` and `COMPLIANCE` mode and its
`retain_until` (±1 s), and non-empty request ids, never `fixture-…`;
`listing.prefix` the run root, `listing.keys` exactly the locked keys, and
a live request id.

The L4 gates are `test_evidence_every_probe_has_a_live_result`,
`test_evidence_cleanup_left_only_locked_objects`,
`test_evidence_was_verified_against_the_live_bucket` and
`test_evidence_bucket_had_no_default_retention`.

## The decision record `decisions/STORAGE-LANE-D.md`

The builder ships a template. Offline (always) it must have exactly one line
starting `Chosen path:` and exactly one starting `Evidence sha256:` (an
optional list marker and `**` bold are allowed), name all seven probes and
the three path ids (`presigned_multipart`, `single_presigned_put`,
`upload_relay`), and point at `evidence/w3a_probe.json`.

With `$LANED_EVIDENCE` set, the `Chosen path:` value (backticks and `*`
stripped) must be **exactly one** path id, supported by the evidence
(`presigned_multipart` needs `presigned_upload_part: pass`,
`single_presigned_put` needs `single_put_limit: pass`, `upload_relay` is
always available), and `Evidence sha256:` must be the sha256 of the evidence
file's bytes.

Owner order: run the probe; run `storage_probe verify --evidence
evidence/w3a_probe.json` the same day (before the locks expire); then fill
`Chosen path:` and `Evidence sha256:` (`sha256sum evidence/w3a_probe.json`,
of the verified file) in an implementation-mode PR touching only the
decision record; after it is on main, open the evidence PR adding only
`evidence/w3a_probe.json`.

Run (prod box, tests only, offline; `PYTHONPATH=<worktree>` so an editable
install doesn't shadow the worktree):
`nice -n 19 env PYTHONPATH=<wt> /root/cascade-v2/.venv/bin/python -P -m pytest --rootdir <wt> --noconftest -p no:cacheprovider -q -m 'not torch' <wt>/tests/accept/w3a/`

The in-process tests (`test_w3a_default_retention.py`,
`test_w3a_live_verify.py`) import `cascade.laned.storage_probe` with every
socket connect refused and `boto3`/`botocore` unimportable, and hand it a
`helpers.FakeS3` client; nothing reaches Hippius.
