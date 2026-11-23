---
id: STORAGE-LANE-D
type: decision
title: "Lane D vault: the miner upload path, chosen from the W3a live Hippius probe"
status: accepted
date: 2026-11-23
tags: [lane-d, storage, hippius, vault]
relations: {plan: "docs/plans/LANE_D_PLAN.md §1.2 Storage, §3.0 W3a", unblocks: [W3, W6d, W10, W13, W18]}
---

# Storage decision: lane D upload path

This record names the **one** upload path the lane D intake (W3) and the miner
CLI (W10) build. It is chosen by the owner from the W3a live probe evidence,
`evidence/w3a_probe.json`, never from documentation alone.
`scripts/laned/accept W3a --evidence evidence/w3a_probe.json` checks that the
choice below is a path that evidence supports and that the sha256 below is the
evidence file's.

## Decision

Chosen path: presigned_multipart

Evidence sha256: 9298ff34af7f229d44cee99ef06d9e4268fbdd4f7d7b01519f7add7483202a2c

The sha256 is of the verified evidence file (`verified_live.verified: true`,
`binding: unlocked_anchor`, verified 2026-11-23T04:52:15Z). The file is added
to main by a separate evidence-only PR after this record lands.

## Path ids

| Path id | What the miner does | Needs (W3a probe result) |
|---|---|---|
| `presigned_multipart` | Multipart upload; the intake mints one short-lived presigned UploadPart URL per part (≤ 512 MiB parts, ≤ 10,000 parts) and completes the upload itself | `presigned_upload_part: pass` |
| `single_presigned_put` | One presigned PUT per file; the ~2.5 GB BF16 weights in one request | `single_put_limit: pass` |
| `upload_relay` | A small upload relay off the box receives the miner's stream and writes the vault with the vault credential | always available (fallback) |

The probe's `recommended_path` is the first supported row of this table. The
owner may choose a later supported row and says why under "Rationale".

## Probe results (copy from the evidence)

| Probe | Result | Note |
|---|---|---|
Run `f466f7eb48fc10a9`, `s3.hippius.com`, bucket `cascade-w3a-probe-20261122`,
prefix `w3a-probe/`, 2026-11-23 04:45:27–04:51:47 UTC, `--no-object-lock`
(`bucket_lock: {"lock_enabled": false, "asserted_by": "owner_flag"}`). Results
are verbatim from the evidence (`result` / `reason`).

| Probe | Result | Note |
|---|---|---|
| `presigned_upload_part` | `pass` | the capability behaves as lane D needs. Two presigned parts (5 MiB + 1 KiB), 200 + ETag each, no `Authorization` header; complete 200; object 5,243,904 bytes, sha256 matches the upload |
| `single_put_limit` | `error` | `BrokenPipeError: [Errno 32] Broken pipe` on the 2,684,354,560-byte presigned PUT. `max_accepted_bytes` 536,870,912 (512 MiB PUT: 200, size and sha256 match) / `min_rejected_bytes` null (the 2.5 GiB PUT got no HTTP answer: the connection was dropped) |
| `object_lock_multipart` | `skipped` | `no_object_lock` (owner flag): lock headers on CreateMultipartUpload are not testable, Hippius has no Object Lock |
| `prefix_bucket_policy` | `error` | `RuntimeError: GetBucketPolicy answered 403 (AccessDenied): cannot tell whether a policy exists, so it is not touched`. No policy was put; prefix-scoped grants (§1.2) are unproven |
| `subtoken_scope` | `error` | `no conclusive result: see the observations`. `HeadBucket(other,main)` 403, `other_bucket_exists: false`; in-bucket put/get and other-bucket list never ran. Inconclusive (see below) |
| `upload_part_copy_2_5gb` | `pass` | the capability behaves as lane D needs. 2,684,354,560-byte source; single CopyObject 200; 10 × 256 MiB UploadPartCopy parts, contiguous, each 200; complete 200; dest size and sha256 equal the source |
| `compliance_retention_unshortenable` | `skipped` | `no_object_lock` (owner flag) |

Top level: `recommended_path: presigned_multipart`, `passed: false` (two
probes ended in `error`), `locked_objects: []`, cleanup deleted 5 versions
and aborted 1 upload, `remaining_unlocked: []`. The run root holds only the
unlocked anchor `w3a-probe/f466f7eb48fc10a9/anchor.bin` (1024 bytes), which
`storage_probe verify` re-read (`binding: unlocked_anchor`).

**Sub-token scope is inconclusive, and why.** The main Hippius key has no
access to the owner-created throwaway bucket, so the owner's full-access key
for that bucket was used in **both** roles (main key and sub-token). The
main-role HeadBucket on `--other-bucket` (`cascade-manifests`) therefore
answered 403 and the probe could not establish that the other bucket exists,
so it stopped without testing scope. The evidence says nothing either way
about whether a Hippius sub-token is bucket-scoped.

**Single PUT is unsupported at king size.** A 512 MiB presigned PUT works; the
~2.5 GB BF16 weights in one PUT broke the connection. `single_presigned_put`
is not a supported path.

## Consequences that follow from the results

- **No Object Lock:** Hippius has no Object Lock (owner decision rev6.4), so kings are not locked by copy at crowning; immutability comes from the content-addressed path (the manifest digest in the key) + the ingest re-hash (`digest_mismatch`) + the signed report digest + the R2/HF mirrors.
- **Crowning copy** (plan §1.2, rev5 as amended by rev6.4): W6d may still
  server-side copy the king to its crowned path; `upload_part_copy_2_5gb:
  pass` shows a ~2.5 GB UploadPartCopy works on Hippius with the bytes
  intact. The copy is plain (no retention); it protects nothing by itself.
- **Pending entries are never locked**: content addressing (the manifest
  digest in the path) plus the ingest re-hash stop overwrites
  (`digest_mismatch`). With no Object Lock nothing is locked at all, so TTL
  deletion of losers (D16) is unaffected
  (`compliance_retention_unshortenable: skipped`).
- **Credentials**: the eval pod gets presigned GETs only; miners get
  presigned UploadPart URLs only (`authorization_header_sent: false`), never
  a key. Sub-token scope is unproven (`subtoken_scope: error`, inconclusive
  above), so any sub-token is treated as possibly account-wide: it goes to
  the `pool_d` builder only, never to a pod or a miner. Prefix-scoped
  bucket policies are likewise unproven (`prefix_bucket_policy: error`,
  GetBucketPolicy 403) and nothing relies on them.

## Rationale

`presigned_multipart` is the probe's `recommended_path` and the only
supported path besides the `upload_relay` fallback. `presigned_upload_part:
pass` shows Hippius accepts UploadPart through presigned URLs with no
`Authorization` header and completes the upload byte-exact, so the intake
can hand a miner short-lived per-part URLs and keep the vault credential to
itself. `single_put_limit: error` (the 2.5 GiB PUT broke the connection;
512 MiB was the largest accepted) rules out `single_presigned_put` at king
size, and a relay would put ~2.5 GB per entry through infrastructure we run
for no gain. `upload_part_copy_2_5gb: pass` means a king can still be copied
server-side at crowning. The two lock probes are `skipped (no_object_lock)`
because Hippius has no Object Lock (owner decision rev6.4), so immutability
rests on the content-addressed path, the ingest re-hash, the signed report
digest and the R2/HF mirrors rather than on retention. `prefix_bucket_policy`
and `subtoken_scope` ended in `error` (a GetBucketPolicy 403, and a probe run
with the same full-access key in both roles); neither affects the upload
path, and the credentials rule above assumes the worst for both.

## Live run

- Command, from a clean main checkout with the `HIPPIUS_*` environment set:
  `python -P -m cascade.laned.storage_probe --bucket <throwaway bucket> --prefix <throwaway prefix>/ --other-bucket <bucket outside the sub-token> --out evidence/w3a_probe.json --confirm-live --no-object-lock`
  Hippius has no Object Lock (owner decision rev6.4), so the bucket has
  none and `--no-object-lock` asserts that: the probe makes no lock read,
  sends no lock header and skips the two lock probes. (Without the flag the
  fail-closed lock preflight refuses every Hippius bucket,
  `bucket_lock_unreadable`.)
- Verify, the same day (read-only; same environment):
  `python -P -m cascade.laned.storage_probe verify --evidence evidence/w3a_probe.json`
  With no locked objects it re-reads the unlocked anchor object, lists the
  run root, and adds the hash-bound `verified_live` record
  (`binding: unlocked_anchor`) to the file. Only then run
  `sha256sum evidence/w3a_probe.json` for the Evidence sha256 above (of the
  verified file), and never edit the file afterwards.
- This run: run id `f466f7eb48fc10a9`, endpoint host `s3.hippius.com`,
  bucket `cascade-w3a-probe-20261122`, prefix `w3a-probe/`; locked objects:
  none. Left in the bucket: the unlocked anchor
  `w3a-probe/f466f7eb48fc10a9/anchor.bin` (1024 bytes), which the owner may
  delete after the evidence PR. Verified 2026-11-23T04:52:15Z.
