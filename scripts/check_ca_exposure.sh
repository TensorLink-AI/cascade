#!/usr/bin/env bash
# Flag any endpoint cascade talks to whose served TLS chain runs through an
# emSign / eMudhra CA (2026-10 intermediate-key compromise: 6 TLS + 2
# code-signing intermediates). Run from the orchestrator / validator hosts —
# the result depends on the egress path, so a sandbox behind an intercepting
# proxy tells you nothing.
#
#   scripts/check_ca_exposure.sh [extra-host ...]
#
# Exit 1 if any host's chain names emSign/eMudhra, 2 if any host was
# unreachable, else 0.
set -u

HOSTS=(
  s3.hippius.com registry.hippius.com hub.hippius.com
  us-east-1.hippius.com eu-central-1.hippius.com
  huggingface.co submissions.cascadesub.net lium.io api.shadeform.ai
  ghcr.io github.com download.pytorch.org pypi.org files.pythonhosted.org
  "$@"
)
PATTERN='emsign|emudhra'

hit=0 unreachable=0
for h in "${HOSTS[@]}"; do
  chain=$(timeout 15 openssl s_client -connect "$h:443" -servername "$h" \
            -showcerts </dev/null 2>/dev/null)
  if [ -z "$chain" ]; then
    printf '%-40s UNREACHABLE\n' "$h"; unreachable=1; continue
  fi
  issuers=$(printf '%s\n' "$chain" | grep -E '^ *[0-9]+ s:|^ +i:' \
              | sed -E 's/^ +//' | paste -sd' ' -)
  if printf '%s' "$issuers" | grep -qiE "$PATTERN"; then
    printf '%-40s EXPOSED  %s\n' "$h" "$issuers"; hit=1
  else
    leaf_issuer=$(printf '%s\n' "$chain" | grep -m1 -E '^ +i:' | sed -E 's/^ +i://')
    printf '%-40s ok       %s\n' "$h" "$leaf_issuer"
  fi
done

# Local trust stores: which emSign roots does this host still trust?
echo
echo "emSign roots still in local trust stores:"
ls /etc/ssl/certs 2>/dev/null | grep -iE "$PATTERN" | grep -v '^[0-9a-f]\{8\}\.' \
  | sed 's/^/  system: /'
certifi_pem=$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null)
[ -f "$certifi_pem" ] && grep -iE "^# Label:.*($PATTERN)" "$certifi_pem" \
  | sed 's/^/  certifi: /'

[ $hit -eq 1 ] && exit 1
[ $unreachable -eq 1 ] && exit 2
exit 0
