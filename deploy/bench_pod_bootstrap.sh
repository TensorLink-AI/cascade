#!/usr/bin/env bash
# Bench-pod bootstrap for the validator's elastic bench lease
# (cascade/validator/bench_lease.py, `cascade-validator --bench-lease`).
#
# Runs ON the freshly rented pod over SSH, as the lease's `bootstrap_command`.
# Idempotent: a re-run on a pod that is already set up only re-verifies.
# Exit 0 = the pod can sweep; anything else = the lease terminates it and
# re-rents after its backoff (the settlement holds meanwhile — never a verdict).
#
# Produces exactly the standing bench pod's layout (valeval-blackwell3):
#   <workdir>/.venv            validator-side helpers (bench_identity) — from the image
#   <workdir>/benchmarks/.venv the sidecar, `uv sync --frozen` from benchmarks/uv.lock
#   <data>/{gift-eval,boom,time,fev}  the pinned benchmark data (~5 GB)
set -euo pipefail

WORKDIR="${CASCADE_WORKDIR:-/root/cascade}"
DATA="${CASCADE_BENCH_DATA:-/data/bench}"
SUITES="${CASCADE_BENCH_SUITES:-gift-eval,boom,time,fev}"

log() { printf '%s bench-bootstrap: %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }

cd "$WORKDIR"
[ -x "$WORKDIR/.venv/bin/python" ] || { log "no $WORKDIR/.venv (wrong image?)"; exit 3; }
command -v uv >/dev/null || { log "uv missing from the image"; exit 3; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >&2 \
  || { log "no GPU visible"; exit 4; }

log "sidecar env (benchmarks/uv.lock, frozen)"
(cd benchmarks && uv sync --frozen --extra time --extra fev >&2)
benchmarks/.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"' \
  || { log "sidecar torch cannot see the GPU"; exit 4; }

mkdir -p "$DATA"
log "benchmark data -> $DATA ($SUITES)"
benchmarks/.venv/bin/cascade-benchmark-download --data-dir "$DATA" --suites "$SUITES" >&2

# The same identity check the validator runs before reusing a cached side: a
# partial or unpinned download must fail HERE, not inside a sweep.
ident="$("$WORKDIR/.venv/bin/python" -m cascade.validator.bench_identity --data-dir "$DATA")"
case "$ident" in
  null|"") log "benchmark data identity unresolved: $ident"; exit 5 ;;
esac
log "ready: $ident"
