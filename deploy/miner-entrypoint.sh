#!/usr/bin/env bash
# cascade-miner image entrypoint.
#
#   docker run … cascade-miner                 → the web UI (cascade mine-ui) on :8765
#   docker run … cascade-miner mine --…        → any `cascade` subcommand, headless
#   docker run … cascade-miner bash            → a shell
set -euo pipefail

mkdir -p /work
cd /work

case "${1:-ui}" in
  ui)
    shift || true
    exec cascade mine-ui --host 0.0.0.0 --port "${PORT:-8765}" \
      --workdir "${CASCADE_MINE_WORKDIR:-/work/mine-run}" "$@"
    ;;
  bash|sh|claude)
    exec "$@"
    ;;
  *)
    exec cascade "$@"
    ;;
esac
