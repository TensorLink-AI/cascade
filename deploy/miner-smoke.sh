#!/usr/bin/env bash
# Smoke-test a built cascade-miner image BEFORE it is pushed (CI) or used.
#
#   deploy/miner-smoke.sh <image> [toolbox|oneclick]
#
# Exercises the real entrypoint, CLI, admission checks, the mine loop with the
# shipped example strategy (scoring stubbed: a real score is GPU-minutes), the
# UI, and for `oneclick` the Claude Code CLI. Exits non-zero on the first failure.
set -euo pipefail

IMG="${1:?usage: miner-smoke.sh <image> [toolbox|oneclick]}"
TARGET="${2:-toolbox}"
WORK="$(mktemp -d)"
UI_NAME="cascade-miner-smoke-$$"
trap 'docker rm -f "$UI_NAME" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT
run() { docker run --rm -v "$WORK:/work" "$@"; }
step() { printf '\n== %s\n' "$*"; }

step "entrypoint help"
run "$IMG" help | grep -q "cascade-miner: the cascade miner CLI"

step "build provenance"
run "$IMG" python -c 'import os; s = os.environ.get("CASCADE_MINER_BUILD_SHA", ""); print(s); assert s and s != "unknown"'

step "CLI subcommands parse"
for sub in verify score fetch mine ralph submit mine-ui; do
  run "$IMG" "$sub" --help >/dev/null
done

step "verify: example generator (admission checks + determinism)"
run "$IMG" verify /opt/cascade/scripts/example_generator | grep -q "OK: generator would be accepted"

step "mine --proposer cmd with the shipped example strategy (scoring stubbed)"
cat > "$WORK/smoke_mine.py" <<'EOF'
import json, shutil
from pathlib import Path
from cascade.miner import optimize as opt
from cascade.shared.config import load_chain_config
gen = Path("/work/gen")
shutil.copytree("/opt/cascade/scripts/example_generator", gen, dirs_exist_ok=True)
(gen / "config.json").write_text(json.dumps({"family_weights": {"a": 0.2, "b": 0.8}}))
cfg = opt.LoopConfig(workdir=Path("/work/mine-run"), start_dir=gen, proposer="cmd",
                     iterations=1, propose_cmd="python /opt/cascade/scripts/example_strategy.py")
loop = opt.OptimizationLoop(cfg, chain_cfg=load_chain_config(), score_fn=lambda d, s:
                            1 - json.loads((d / "config.json").read_text())["family_weights"]["a"])
st = loop.run()
h = opt.read_history("/work/mine-run")
assert st["status"] == "finished", st
assert [r["status"] for r in h] == ["baseline", "scored"], h
assert h[1]["accepted"] and h[1]["note"] == "cmd: family_weights.a x2", h[1]
print("loop ok:", [(r["iteration"], r["status"], round(r["score"], 3)) for r in h])
EOF
run "$IMG" python /work/smoke_mine.py

step "ralph preflight fails cleanly without a key"
if run -e CASCADE_NO_DOTENV=1 "$IMG" ralph --llm-provider chutes --llm-model m --check 2>"$WORK/err"; then
  echo "expected a missing-key error"; exit 1
fi
grep -q 'CHUTES_API_KEY is not set' "$WORK/err"

step "UI boots and answers with the token"
docker run -d --name "$UI_NAME" -p 127.0.0.1::8765 -e CASCADE_UI_TOKEN=smoke \
  -v "$WORK:/work" "$IMG" ui >/dev/null
PORT="$(docker port "$UI_NAME" 8765/tcp | head -1 | sed 's/.*://')"
for _ in $(seq 1 30); do
  curl -fsS -o /dev/null -H "X-Cascade-Token: smoke" "http://127.0.0.1:$PORT/api/status" && break
  sleep 1
done
curl -fsS -H "X-Cascade-Token: smoke" "http://127.0.0.1:$PORT/api/status" \
  | python3 -c 'import json, sys; d = json.load(sys.stdin); assert "history" in d and "env" in d'
test "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/status")" = 403

if [ "$TARGET" = "oneclick" ]; then
  step "oneclick: Claude Code + skill present"
  run "$IMG" claude --version
  run "$IMG" bash -c 'test -f /root/.claude/skills/cascade-mine/SKILL.md'
fi

printf '\nsmoke OK: %s (%s)\n' "$IMG" "$TARGET"
