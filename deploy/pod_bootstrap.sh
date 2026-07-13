#!/usr/bin/env bash
# Bootstrap a freshly rented bare GPU pod into a cascade worker over SSH.
#
# Run BY the provisioner ON the orchestrator ([provisioner] bootstrap_script);
# pod coordinates arrive via env: POD_IP POD_PORT POD_USER POD_KEY POD_STAGE
# POD_WORKDIR. Idempotent — safe to re-run on a half-bootstrapped pod.
#
# What it does: push the orchestrator's source tree (no venv, no git, no
# work dirs), install uv, and `uv sync --frozen` against the PINNED lock
# (python 3.11 + torch 2.4.1+cu124 — the reproducibility contract). Creds are
# NOT seeded: the trainer forwards HIPPIUS_*/HF_TOKEN per dispatch.
set -euo pipefail

: "${POD_IP:?}" "${POD_PORT:?}" "${POD_USER:?}" "${POD_KEY:?}" "${POD_WORKDIR:?}"
SRC_ROOT="${CASCADE_SRC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SSH_OPTS=(-p "$POD_PORT" -i "$POD_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
DEST="$POD_USER@$POD_IP"

rsync -a --delete-after \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '_train_work' --exclude 'bench_data' --exclude '*.log' \
  -e "ssh ${SSH_OPTS[*]}" \
  "$SRC_ROOT/" "$DEST:$POD_WORKDIR/"

# The live deployment's chain toml can carry uncommitted testnet overrides
# (window_pool, budget knobs) that fold into contract_digest — push the REAL
# one when the operator points at it, or the pod would sign a different digest.
if [[ -n "${CASCADE_CHAIN_TOML:-}" ]]; then
  scp -P "$POD_PORT" -i "$POD_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new     "$CASCADE_CHAIN_TOML" "$DEST:$POD_WORKDIR/$(basename "$CASCADE_CHAIN_TOML")"
fi

ssh "${SSH_OPTS[@]}" "$DEST" bash -s <<REMOTE
set -euo pipefail
cd "$POD_WORKDIR"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
uv sync --frozen --no-dev
.venv/bin/python -c 'import sys, torch, cascade.trainer.worker; print("bootstrap ok:", sys.version.split()[0], torch.__version__)'
REMOTE
