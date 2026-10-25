#!/usr/bin/env bash
# Bootstrap a bare worker over SSH. The orchestrator supplies a patched uv;
# remote PATH and curl installers are never used. Dotenv files are not copied.
# POD_* arrive from the provisioner. CASCADE_UV_BIN optionally selects uv.
set -euo pipefail

: "${POD_IP:?}" "${POD_PORT:?}" "${POD_USER:?}" "${POD_KEY:?}" "${POD_WORKDIR:?}"
SRC_ROOT="${CASCADE_SRC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
UV_BIN="${CASCADE_UV_BIN:-$(command -v uv)}"
# Reject old installers before transferring or installing anything. Require a
# stable release: prereleases do not establish that the security fix is present.
UV_VERSION="$("$UV_BIN" --version)"
if [[ ! "$UV_VERSION" =~ ^uv\ ([0-9]+)\.([0-9]+)\.([0-9]+)(\ |$) ]]; then
  echo 'bootstrap requires a stable uv >=0.13.4' >&2; exit 1
fi
if (( 10#${BASH_REMATCH[1]} == 0 && (10#${BASH_REMATCH[2]} < 13 || (10#${BASH_REMATCH[2]} == 13 && 10#${BASH_REMATCH[3]} < 4)) )); then
  echo 'bootstrap requires uv >=0.13.4 (patched TLS trust)' >&2; exit 1
fi
UV_SHA="$(sha256sum "$UV_BIN")"
UV_SHA="${UV_SHA%% *}"
SSH_OPTS=(-p "$POD_PORT" -i "$POD_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
DEST="$POD_USER@$POD_IP"

rsync -a --delete-after \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '.env' --exclude '.env.*' --exclude '.bootstrap' \
  --exclude '_train_work' --exclude 'bench_data' --exclude '*.log' \
  -e "ssh ${SSH_OPTS[*]}" \
  "$SRC_ROOT/" "$DEST:$POD_WORKDIR/"

if [[ -n "${CASCADE_CHAIN_TOML:-}" ]]; then
  scp -P "$POD_PORT" -i "$POD_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    "$CASCADE_CHAIN_TOML" "$DEST:$POD_WORKDIR/$(basename "$CASCADE_CHAIN_TOML")"
fi

if [[ "${POD_STAGE:-}" == "eval" && -d "$SRC_ROOT/bench_data" ]]; then
  rsync -a -e "ssh ${SSH_OPTS[*]}" "$SRC_ROOT/bench_data/" "$DEST:$POD_WORKDIR/bench_data/"
fi

printf -v REMOTE_MKDIR 'mkdir -p -- %q' "$POD_WORKDIR/.bootstrap"
ssh "${SSH_OPTS[@]}" "$DEST" "$REMOTE_MKDIR"
scp -P "$POD_PORT" -i "$POD_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
  "$UV_BIN" "$DEST:$POD_WORKDIR/.bootstrap/uv"
# SSH joins command arguments through a remote shell; quote them explicitly.
printf -v REMOTE_COMMAND 'bash -s -- %q %q %q' "$POD_WORKDIR" "$UV_SHA" "${POD_STAGE:-}"
ssh "${SSH_OPTS[@]}" "$DEST" "$REMOTE_COMMAND" <<'REMOTE'
set -euo pipefail
cd "$1"
printf '%s  .bootstrap/uv\n' "$2" | sha256sum -c -
chmod 700 .bootstrap/uv
# Keep uv on PATH for uv-run subprocesses too, without trusting an older uv.
export PATH="$PWD/.bootstrap:$PATH"
# A system/custom-store override would bypass uv's patched bundled CA store.
# Refuse those options during bootstrap; application overrides are checked below.
if [[ -n "${UV_NATIVE_TLS:-}" || -n "${UV_SYSTEM_CERTS:-}" || -n "${SSL_CERT_FILE:-}" || -n "${SSL_CERT_DIR:-}" || -n "${UV_INSECURE_HOST:-}" ]]; then
  echo 'bootstrap requires uv bundled TLS trust without custom/insecure overrides' >&2; exit 1
fi
# The generator sandbox needs unprivileged user/network namespaces.
sysctl -qw kernel.apparmor_restrict_unprivileged_userns=0 2>/dev/null || true
sysctl -qw kernel.unprivileged_userns_clone=1 2>/dev/null || true
# Check OS trust before uv invokes git for direct-reference dependencies.
.bootstrap/uv run --no-project --python 3.11 python scripts/check_tls_trust.py --system-only
.bootstrap/uv sync --frozen --all-extras --no-dev
.venv/bin/python scripts/check_tls_trust.py
.venv/bin/python -c 'import sys, torch, cascade.trainer.worker; print("bootstrap ok:", sys.version.split()[0], torch.__version__)'
# Benchmark dependencies are needed even when no local dataset cache exists.
# Installation and trust-check failures must fail provisioning, not disappear.
if [[ "$3" == "eval" ]]; then
  .bootstrap/uv sync --project benchmarks --frozen --extra time --extra fev
  benchmarks/.venv/bin/python scripts/check_tls_trust.py
fi
REMOTE
