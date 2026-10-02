#!/usr/bin/env bash
# Wake Hermes every OPERATOR_INTERVAL_SECONDS to read /operator/status.json and
# steer the gauntlet through /operator/DIRECTIVES.md. One fresh session per wake
# (state lives in /operator files, not in a long-lived context).
set -euo pipefail

: "${HERMES_MODEL:?set HERMES_MODEL (a model id your endpoint serves)}"
: "${HERMES_BASE_URL:?set HERMES_BASE_URL (OpenAI-compatible root, e.g. https://llm.chutes.ai/v1)}"
: "${HERMES_API_KEY:?set HERMES_API_KEY (the endpoint key; env only)}"

# Hermes host-gates env keys (OPENAI_API_KEY only ever goes to openai.com), so a
# custom endpoint's key goes in model.api_key. The file stays in this
# container's home, never in the mounted /operator.
mkdir -p ~/.hermes /operator/reports
umask 077
cat > ~/.hermes/config.yaml <<EOF
model:
  default: ${HERMES_MODEL}
  provider: custom
  base_url: ${HERMES_BASE_URL}
  api_key: ${HERMES_API_KEY}
EOF
umask 022

INTERVAL="${OPERATOR_INTERVAL_SECONDS:-7200}"
while true; do
  if [[ -f /operator/status.json ]]; then
    stamp="$(date -u +%Y-%m-%dT%H%MZ)"
    echo "== ${stamp} operator wake" >> /operator/reports/operator.log
    # --yolo: no TTY to answer approval prompts; the container boundary (only
    # /operator mounted, no keys but its own) is the sandbox.
    if ! timeout "${OPERATOR_TIMEOUT_SECONDS:-1800}" \
        hermes chat -Q --yolo -s cascade-operator --max-turns "${OPERATOR_MAX_TURNS:-40}" \
        --source tool -q "$(cat /opt/operator/OPERATOR_PROMPT.md)" \
        >> /operator/reports/operator.log 2>&1; then
      echo "== ${stamp} operator run failed (see above)" >> /operator/reports/operator.log
    fi
  fi
  sleep "${INTERVAL}"
done
