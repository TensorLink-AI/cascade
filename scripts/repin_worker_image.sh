#!/usr/bin/env bash
# Re-pin [training] train_image_digest to a published worker image tag.
#
#   scripts/repin_worker_image.sh worker-v0.2.0 [chain.toml]
#
# Resolves the tag's manifest digest from GHCR (anonymous pull token) and
# rewrites the pin in the given toml (default chain.toml). The digest folds
# into contract_digest, so deploying the change requires the coordinated
# restart protocol printed at the end.
set -euo pipefail
TAG="${1:?usage: repin_worker_image.sh <tag> [chain.toml]}"
TOML="${2:-chain.toml}"
REPO="tensorlink-ai/cascade-worker"

TOKEN=$(curl -sf "https://ghcr.io/token?scope=repository:${REPO}:pull" | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])")
DIGEST=$(curl -sfI -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.v2+json" \
  "https://ghcr.io/v2/${REPO}/manifests/${TAG}" | grep -i docker-content-digest | awk '{print $2}' | tr -d '\r')
[[ "$DIGEST" == sha256:* ]] || { echo "could not resolve digest for tag ${TAG}" >&2; exit 1; }

python3 - "$TOML" "$DIGEST" <<'PY'
import re, sys
toml, digest = sys.argv[1], sys.argv[2]
s = open(toml).read()
s2, n = re.subn(r'train_image_digest = "sha256:[0-9a-f]{64}"',
                f'train_image_digest = "{digest}"', s, count=1)
if n != 1:
    sys.exit(f"expected exactly one pinned train_image_digest in {toml}; found {n}")
open(toml, "w").write(s2)
print(f"pinned {digest} in {toml}")
PY

cat <<PROTO

Re-pin deploy protocol (the digest folds into contract_digest):
  1. commit + merge this toml change
  2. at an epoch boundary: restart trainer AND all validators together
     (a digest mismatch rejects every manifest until both sides agree)
  3. pods: fresh rentals pull the image automatically; static pods must
     pull ${DIGEST} before the next round
PROTO
