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

# 2026-11-27: the eval caches go on the pod's large data mount (a 43 GB root
# overlay filled while /workspace sat empty). A missing / empty cache dir
# becomes a symlink into the largest writable mount on another filesystem
# (CASCADE_DATA_MOUNTS, default /workspace); a non-empty one is left alone.
# Twin of cascade/shared/pod_data_root.py:place_on_data_mount_command.
avail() { df -PB1 "$1" 2>/dev/null | awk 'NR==2{print $4}'; }
for name in _eval_offload _private_eval_offload _lane_b_offload; do
  L="$WORKDIR/$name"
  if [ -d "$L" ] && [ ! -L "$L" ]; then rmdir "$L" 2>/dev/null || true; fi
  [ -e "$L" ] || [ -L "$L" ] && continue
  best=""; bestf="$(avail "$WORKDIR")"
  for c in ${CASCADE_DATA_MOUNTS:-/workspace}; do
    [ -d "$c" ] && [ -w "$c" ] || continue
    [ "$(stat -c %d "$c")" = "$(stat -c %d "$WORKDIR")" ] && continue
    f="$(avail "$c")"
    [ "${f:-0}" -gt "${bestf:-0}" ] && { best="$c"; bestf="$f"; }
  done
  if [ -n "$best" ]; then
    mkdir -p "$best/$name" && ln -sfn "$best/$name" "$L"
    log "$name -> $best/$name ($((bestf / 1000000000)) GB free)"
  fi
done

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
