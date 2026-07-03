#!/usr/bin/env bash
# Make the receipts/* objects of the testnet manifest bucket public-read so
# third parties can run `cascade-audit` with zero credentials. Uses per-OBJECT
# canned ACLs (Hippius supports them; see docs.hippius.com/storage/s3) so the
# rest of the bucket — manifests, logs — stays private. New receipts are
# published with the ACL automatically (shared/hippius.publish_receipt); this
# script backfills the ones already written. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a && . ./.env && set +a
.venv/bin/python - <<'EOF'
from cascade.shared.config import load_chain_config
from cascade.shared.hippius import S3Config, S3Store

cfg = load_chain_config("chain.testnet.toml")
b = cfg.storage.manifest_bucket
c = S3Store(S3Config.from_storage(cfg.storage, bucket=b)).client()
keys = [o["Key"] for page in c.get_paginator("list_objects_v2").paginate(
            Bucket=b, Prefix="receipts/") for o in page.get("Contents", [])]
print(f"{len(keys)} receipt object(s) in {b}")
for k in keys:
    try:
        c.put_object_acl(Bucket=b, Key=k, ACL="public-read")
        print(f"  public-read: {k}")
    except Exception as e:
        print(f"  FAILED {k}: {type(e).__name__}: {e}")
EOF
echo -n "anonymous probe: HTTP "
curl -s -o /dev/null -w "%{http_code}\n" \
  "https://s3.hippius.com/cascade-testnet-manifests/receipts/latest.json"
