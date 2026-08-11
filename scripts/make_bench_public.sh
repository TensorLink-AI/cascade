#!/usr/bin/env bash
# Make the benchmarks/* objects of a manifest bucket public-read so the
# stakeholder scoreboard can read them anonymously from the browser.
#
# Why this exists: until the ACL fix in shared/bench_report.publish_bench_report,
# the trainer's post-publish bench wrote benchmarks/round-<id>.json PRIVATE
# (every other browser-read object — receipts, status/*, the pages — carried the
# canned ACL). A private round doc reads 403, which the page cannot tell apart
# from "never benched": the Public benchmark cell just stays dashed. New reports
# now publish public-read automatically; this backfills the ones already written.
#
# Per-OBJECT canned ACLs (Hippius supports them; see docs.hippius.com/storage/s3)
# so the rest of the bucket — manifests, logs — stays private. Safe to re-run.
#
# Usage:  scripts/make_bench_public.sh [chain.toml]     (default: chain.toml)
set -euo pipefail
cd "$(dirname "$0")/.."
CHAIN_TOML="${1:-chain.toml}"
set -a && . ./.env && set +a
exec .venv/bin/python - "$CHAIN_TOML" <<'EOF'
import sys
import urllib.error
import urllib.request

from cascade.shared.config import load_chain_config
from cascade.shared.hippius import S3Config, S3Store

cfg = load_chain_config(sys.argv[1])
bucket = cfg.storage.manifest_bucket
s3cfg = S3Config.from_storage(cfg.storage, bucket=bucket)
c = S3Store(s3cfg).client()

keys = [o["Key"] for page in c.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix="benchmarks/") for o in page.get("Contents", [])]
print(f"{len(keys)} benchmark object(s) in {bucket}")
failed = 0
for k in keys:
    try:
        c.put_object_acl(Bucket=bucket, Key=k, ACL="public-read")
        print(f"  public-read: {k}")
    except Exception as e:  # noqa: BLE001 — report every failure, exit non-zero once
        failed += 1
        print(f"  FAILED {k}: {type(e).__name__}: {e}")

# Verify the way the page does: an ANONYMOUS read of one object (a bucket
# listing stays private, so probing the prefix would 403 either way).
if keys:
    url = f"{s3cfg.endpoint.rstrip('/')}/{bucket}/{keys[-1]}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 — fixed https endpoint
            print(f"anonymous probe: HTTP {r.status} {url}")
    except urllib.error.HTTPError as e:
        print(f"anonymous probe: HTTP {e.code} {url}")
        failed += 1
    except OSError as e:
        print(f"anonymous probe FAILED ({type(e).__name__}: {e}) {url}")
        failed += 1

sys.exit(1 if failed else 0)
EOF
