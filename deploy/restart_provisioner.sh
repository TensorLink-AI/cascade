#!/bin/bash
# One-shot provisioner relaunch — exact prior arg line (2026-08-21 v0.6.0 go-live).
# Live copy on the box: /root/cascade/restart_provisioner.sh (replace on deploy).
#
# Hardened after the r47 stray-trio incident (OPSLOG 2026-08-28 12:35): a
# stale provisioner survived the 07:41 flip (pkill pattern vs relative-path
# cmdlines) and double-rented final pods. See restart_trainer.sh for the full
# rationale; the mechanism here is identical:
#   enumerate ALL shapes -> TERM -> wait -> KILL -> refuse to launch if any
#   survive -> launch -> verify the only matching pid is the new one (strays
#   fail LOUD, exit 1, never auto-killed).
# REMINDER (operational invariant): never restart the provisioner inside its
# pre-boundary trigger window.
#
# Dry run (no kill, no launch — prints what would happen):
#   DRY_RUN=1 ./restart_provisioner.sh    or    ./restart_provisioner.sh --dry-run
cd /root/cascade || exit 1

SERVICE="cascade-provisioner"
LAUNCH_DESC=".venv/bin/python .venv/bin/cascade-provisioner --config provision.toml ..."
# Console entrypoint under any path prefix + python -m module forms
# (cascade.provision / cascade.provision.main).
PATTERNS=(
  'cascade-provisioner'
  '-m cascade\.provision(\.main)?([[:space:]]|$)'
)

DRY_RUN="${DRY_RUN:-0}"
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# Union of pids matching any shape, excluding this script's own shell and its
# parent (so the script can never kill itself).
collect_pids() {
  local pat pids=""
  for pat in "${PATTERNS[@]}"; do
    pids="$pids $(pgrep -f -- "$pat" || true)"
  done
  echo "$pids" | tr ' ' '\n' | sed '/^$/d' | sort -un \
    | grep -v -x -e "$$" -e "$PPID" || true
}

OLD_PIDS="$(collect_pids)"

if [ "$DRY_RUN" = "1" ]; then
  if [ -n "$OLD_PIDS" ]; then
    echo "[dry-run] would TERM (then KILL if needed) these $SERVICE pids:"
    ps -o pid,lstart,cmd -p $OLD_PIDS
  else
    echo "[dry-run] no running $SERVICE instances found"
  fi
  echo "[dry-run] would then launch: $LAUNCH_DESC"
  exit 0
fi

if [ -n "$OLD_PIDS" ]; then
  echo "stopping old $SERVICE instance(s):"
  ps -o pid,lstart,cmd -p $OLD_PIDS || true
  kill -TERM $OLD_PIDS 2>/dev/null
  for _ in $(seq 1 30); do
    sleep 1
    OLD_PIDS="$(collect_pids)"
    [ -z "$OLD_PIDS" ] && break
  done
  if [ -n "$OLD_PIDS" ]; then
    echo "TERM did not stick after 30s — sending KILL to: $OLD_PIDS"
    kill -KILL $OLD_PIDS 2>/dev/null
    sleep 2
    OLD_PIDS="$(collect_pids)"
  fi
fi
if [ -n "$OLD_PIDS" ]; then
  echo "FATAL: old $SERVICE instance(s) survived TERM+KILL — NOT launching:"
  ps -o pid,lstart,cmd -p $OLD_PIDS || true
  exit 1
fi

set -a && . ./.env && set +a
test -n "$HF_TOKEN" || { echo "HF_TOKEN missing — aborting"; exit 1; }
nohup .venv/bin/python .venv/bin/cascade-provisioner \
  --config provision.toml --chain-toml chain.toml \
  --work-root /root/cascade/_train_work \
  >> provisioner-service.log 2>&1 &
NEW_PID=$!
echo "provisioner pid $NEW_PID"

# No-strays verification — extra survivors print loudly and exit non-zero;
# deliberately no auto-kill (operator must look first).
sleep 3
STRAYS=""
for pid in $(collect_pids); do
  [ "$pid" = "$NEW_PID" ] && continue
  ppid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
  [ "$ppid" = "$NEW_PID" ] && continue
  STRAYS="$STRAYS $pid"
done
if [ -n "$STRAYS" ]; then
  echo "FATAL: STRAY $SERVICE process(es) survived the restart:$STRAYS"
  ps -o pid,lstart,cmd -p $STRAYS || true
  echo "NOT auto-killing — inspect first: ps -eo pid,lstart,cmd | grep cascade"
  exit 1
fi
if ! kill -0 "$NEW_PID" 2>/dev/null; then
  echo "FATAL: new $SERVICE (pid $NEW_PID) exited right after launch — check provisioner-service.log"
  exit 1
fi
echo "verified: the only running $SERVICE is the new pid $NEW_PID (no strays)"
