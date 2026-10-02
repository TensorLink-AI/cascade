#!/usr/bin/env bash
# cascade-miner image entrypoint. The working directory is /work (your mount).
#
#   docker run … cascade-miner                       → this help (toolbox) / the UI (oneclick)
#   docker run … cascade-miner verify ./my-gen       → any `cascade` subcommand
#   docker run … cascade-miner ui                    → the web UI (cascade mine-ui) on :$PORT
#   docker run … cascade-miner python my_script.py   → python / bash / sh / claude pass through
set -euo pipefail

mkdir -p /work
cd /work

case "${1:-help}" in
  help|-h|--help)
    cat <<'EOF'
cascade-miner: the cascade miner CLI in the pinned pod numeric stack.
Your files live in /work (mount a host dir: -v "$PWD:/work").

  verify ./my-gen                        trainer's admission checks (+ determinism)
  score  ./my-gen --warm-start live      train + score locally (GPU: --device cuda)
  fetch  king --out ./king               the reigning generator, to study or fork
  mine   --start ./my-gen --proposer cmd --propose-cmd "python my_strategy.py"
                                         YOUR strategy in the verify → score → keep loop
  mine   --start ./my-gen                built-in tune loop (config.json weights)
  ralph  --llm-provider chutes|saygm|engy|anthropic --llm-model ID [--check]
                                         Ralph loop: an LLM rewrites the generator code
                                         (oneclick image; key via -e CHUTES_API_KEY …)
  submit ./mine-run/best https://submissions.cascadesub.net \
         --wallet-name W --wallet-hotkey H          (spends the hotkey)
  round | queue | heat | duel | reveal-status       read-only chain / round views
  ui                                     web UI for `mine` (publish -p 127.0.0.1:8765:8765)
  gauntlet run --config harness.toml     the multi-stage harness (docs/GAUNTLET.md;
                                         compose stack: deploy/harness/)
  bash | python …                        a shell / the image's python

Starter files: /opt/cascade/scripts/example_generator,
               /opt/cascade/scripts/example_strategy.py,
               /opt/cascade/champions/king
Docs: /opt/cascade/docs/MINER.md, MINER_DOCKER.md, ONE_CLICK_MINING.md
EOF
    ;;
  ui)
    shift
    exec cascade mine-ui --host 0.0.0.0 --port "${PORT:-8765}" \
      --workdir "${CASCADE_MINE_WORKDIR:-/work/mine-run}" "$@"
    ;;
  bash|sh|python|python3|claude|uv)
    exec "$@"
    ;;
  *)
    exec cascade "$@"
    ;;
esac
