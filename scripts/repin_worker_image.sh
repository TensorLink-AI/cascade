#!/usr/bin/env bash
# Re-pin both training contracts, funded image and source fingerprint together.
#
#   scripts/repin_worker_image.sh worker-v0.2.0 [chain.toml]
#
# Resolves the tag's manifest digest from GHCR (anonymous pull token) and
# rewrites the pin in the given toml (default chain.toml). The digest folds
# into contract_digest, so deploying the change requires the coordinated
# restart protocol printed at the end.
set -euo pipefail
TAG="${1:?usage: repin_worker_image.sh <tag> [chain.toml]}"
[[ "$TAG" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] || { echo "invalid image tag" >&2; exit 1; }
TOML="${2:-chain.toml}"
REPO="tensorlink-ai/cascade-worker"
PYTHON_BIN="${CASCADE_PYTHON:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TOKEN=$(curl -sf "https://ghcr.io/token?scope=repository:${REPO}:pull" | "$PYTHON_BIN" -c "import json,sys; print(json.load(sys.stdin)['token'])")
DIGEST=$(curl -sfI -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.v2+json" \
  "https://ghcr.io/v2/${REPO}/manifests/${TAG}" | grep -i docker-content-digest | awk '{print $2}' | tr -d '\r')
[[ "$DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "could not resolve digest for tag ${TAG}" >&2; exit 1; }

"$PYTHON_BIN" "$SCRIPT_DIR/repin_worker_config.py" "$TOML" "ghcr.io/${REPO}:${TAG}@${DIGEST}"

DECLARED=$(grep -E '^declared_contract_from_block' "$TOML" | tr -dc '0-9')

if [[ -n "${DECLARED:-}" && "${DECLARED}" != "0" ]]; then
cat <<PROTO

Re-pin deploy protocol (DEC-CA-0036 declared contract, armed from block ${DECLARED}):
  1. commit + merge this toml change
  2. restart the TRAINER. That's it — the image digest is a declared term, not
     a locked one, so validators keep scoring across the change with no restart
     and nothing to announce.
  3. pods: fresh rentals pull the image automatically; static pods must
     pull ${DIGEST} before the next round
  4. expect a 'contract-declaration' WARN in cascade-audit naming
     train_image_digest until the fleet's chain.toml catches up. That WARN is
     the change being visible, not a fault.
PROTO
else
cat <<PROTO

Re-pin deploy protocol (the digest folds into contract_digest):
  1. commit + merge this toml change
  2. at an epoch boundary: restart trainer AND all validators together
     (a digest mismatch rejects every manifest until both sides agree)
  3. pods: fresh rentals pull the image automatically; static pods must
     pull ${DIGEST} before the next round

  NOTE: this lockstep step exists only because [scoring]
  declared_contract_from_block is 0 in ${TOML}. Arming it (DEC-CA-0036) makes
  an image re-pin a trainer-only deploy.
PROTO
fi
