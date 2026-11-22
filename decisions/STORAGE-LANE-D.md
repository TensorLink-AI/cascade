---
id: STORAGE-LANE-D
type: decision
title: "Lane D vault: the miner upload path, chosen from the W3a live Hippius probe"
status: proposed
date: 2026-11-22
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

Chosen path: (owner fills after the live run: exactly one path id from the list below)

Evidence sha256: (owner fills: the output of `sha256sum evidence/w3a_probe.json`)

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
| `presigned_upload_part` | (fill) | |
| `single_put_limit` | (fill: `max_accepted_bytes` / `min_rejected_bytes`) | |
| `object_lock_multipart` | (fill) | lock headers on CreateMultipartUpload |
| `prefix_bucket_policy` | (fill) | if `pass`, prefix-scoped grants exist (§1.2) |
| `subtoken_scope` | (fill) | sub-tokens go to the `pool_d` builder only |
| `upload_part_copy_2_5gb` | (fill) | locking a king by copy at crowning |
| `compliance_retention_unshortenable` | (fill) | |

## Consequences that follow from the results

- **Kings are locked by copy at crowning** (plan §1.2, rev5): the king is
  server-side copied into a COMPLIANCE-locked prefix. If
  `upload_part_copy_2_5gb` failed, W6d must re-upload the king instead of
  copying it, and says so in its PR.
- **Pending entries are never locked**: content addressing (the manifest
  digest in the path) plus the ingest re-hash stop overwrites
  (`digest_mismatch`). `compliance_retention_unshortenable: pass` is why:
  a lock can't be shortened, so locking losers would block TTL deletion (D16).
- **Credentials**: the eval pod gets presigned GETs only; sub-tokens are
  bucket-scoped and go to the `pool_d` builder only, never to a pod or a
  miner.

## Rationale

(owner fills: one paragraph, citing the probe results above)

## Live run

- Command, from a clean main checkout with the `HIPPIUS_*` environment set:
  `python -P -m cascade.laned.storage_probe --bucket <throwaway lock-enabled bucket> --prefix <throwaway prefix>/ --other-bucket <bucket outside the sub-token> --out evidence/w3a_probe.json --confirm-live`
- Run id, endpoint host, bucket and the COMPLIANCE-locked objects left until
  their retain-until date: (owner fills from the probe's stdout)
