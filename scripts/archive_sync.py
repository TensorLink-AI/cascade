#!/usr/bin/env python3
"""Pull-based audit archiver: copy immutable round artifacts to the archive bucket.

Copies every ``manifests/round-*.json`` and ``receipts/**/round-*.json`` object
from the production bucket into an archive bucket, skipping keys that already
exist there (append-only, enforced client-side — Hippius has no WORM
primitive). Mutable pointers (``latest.json``) are deliberately NOT archived.

The archive bucket MUST live under a SEPARATE Hippius account: Hippius S3 keys
are account-scoped, so an archive under the production account would be
deletable by the same credentials it is meant to survive. Run this from a box
that holds ONLY the archive account's write keys and (optionally) production
read keys — never the reverse.

Environment:
  SOURCE_S3_ENDPOINT       default https://s3.hippius.com
  SOURCE_S3_BUCKET         e.g. cascade-testnet-manifests
  SOURCE_S3_ACCESS_KEY / SOURCE_S3_SECRET_KEY    (read)
  ARCHIVE_S3_ENDPOINT      default https://s3.hippius.com
  ARCHIVE_S3_BUCKET        e.g. cascade-mainnet-archive
  ARCHIVE_S3_ACCESS_KEY / ARCHIVE_S3_SECRET_KEY  (write — the ONLY place these live)

Usage:
  archive_sync.py [--dry-run] [--prefixes manifests/ receipts/]

Cron (hourly is plenty — rounds are epoch-paced):
  17 * * * * cd /opt/cascade-archiver && ./archive_sync.py >> archive.log 2>&1
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys

import boto3
from botocore.config import Config

IMMUTABLE_PATTERNS = ("manifests/round-*.json", "receipts/*/round-*.json",
                      "receipts/round-*.json")
CFG = Config(connect_timeout=10, read_timeout=60,
             retries={"max_attempts": 3, "mode": "standard"})


def _client(side: str):
    endpoint = os.environ.get(f"{side}_S3_ENDPOINT", "https://s3.hippius.com")
    return boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id=os.environ[f"{side}_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ[f"{side}_S3_SECRET_KEY"],
        config=CFG,
    )


def _list_keys(s3, bucket: str, prefix: str) -> set[str]:
    keys: set[str] = set()
    token: str | None = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        keys.update(o["Key"] for o in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            return keys
        token = resp.get("NextContinuationToken")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be copied; write nothing")
    p.add_argument("--prefixes", nargs="+", default=["manifests/", "receipts/"])
    args = p.parse_args(argv)

    src = _client("SOURCE")
    src_bucket = os.environ["SOURCE_S3_BUCKET"]
    dst = None
    dst_bucket = os.environ.get("ARCHIVE_S3_BUCKET", "")
    if not args.dry_run:
        dst = _client("ARCHIVE")

    copied = skipped = 0
    for prefix in args.prefixes:
        source_keys = {
            k for k in _list_keys(src, src_bucket, prefix)
            if any(fnmatch.fnmatch(k, pat) for pat in IMMUTABLE_PATTERNS)
        }
        have = _list_keys(dst, dst_bucket, prefix) if dst is not None else set()
        for key in sorted(source_keys - have):
            if args.dry_run:
                print(f"WOULD COPY {key}")
                copied += 1
                continue
            body = src.get_object(Bucket=src_bucket, Key=key)["Body"].read()
            dst.put_object(Bucket=dst_bucket, Key=key, Body=body)
            print(f"archived {key} ({len(body)} bytes)")
            copied += 1
        skipped += len(source_keys & have)

    print(f"done: {copied} copied, {skipped} already archived")
    return 0


if __name__ == "__main__":
    sys.exit(main())
