#!/bin/bash
# One-shot trainer relaunch — exact prior arg line (2026-08-19 r29 heat re-run).
# Live copy on the box: /root/cascade/restart_trainer.sh (replace on deploy).
#
# Hardened after the r47 stray-trio incident (OPSLOG 2026-08-28 12:35): the
# 07:41 flip's pkill pattern only matched one cmdline shape, so old instances
# started with RELATIVE paths (`.venv/bin/python .venv/bin/cascade-trainer`)
# survived and ran a parallel round. This script now:
#   1. enumerates old pids across EVERY invocation shape (console script under
#      any path prefix, `python -m` module form) via pgrep -f, unioned;
#   2. TERMs them, waits up to 30s, escalates to KILL, and refuses to launch
#      if anything survives;
#   3. launches the new service only after the old ones are verifiably gone;
#   4. verifies post-launch that the ONLY matching pids are the new service
#      (and its direct children) — extra survivors fail LOUD, exit 1, and are
#      deliberately NOT auto-killed.
#
# Dry run (no kill, no launch — prints what would happen):
#   DRY_RUN=1 ./restart_trainer.sh    or    ./restart_trainer.sh --dry-run
cd /root/cascade || exit 1

SERVICE="cascade-trainer"
LAUNCH_DESC=".venv/bin/python .venv/bin/cascade-trainer --wallet-name main_one ..."
# Every local cmdline shape this service can run as (ERE, matched by pgrep -f
# against the full cmdline):
#   - 'cascade-trainer' catches the console entrypoint under ANY path prefix:
#     cascade-trainer, .venv/bin/cascade-trainer,
#     /root/cascade/.venv/bin/cascade-trainer, python .venv/bin/cascade-trainer
#   - the '-m' pattern catches: python -m cascade.trainer / cascade.trainer.main
# Deliberately does NOT match cascade-train-worker / -m cascade.trainer.worker
# (different service) nor lium/ssh cmdlines carrying cascade-* pod names.
PATTERNS=(
  'cascade-trainer'
  '-m cascade\.trainer(\.main)?([[:space:]]|$)'
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
nohup .venv/bin/python .venv/bin/cascade-trainer \
  --wallet-name main_one --wallet-hotkey sn_validator_hk \
  --network finney --chain-toml chain.toml \
  --trainer cascade.trainer.toto2_trainer:Toto2Trainer \
  --remote-hosts hosts.toml --hosts-wait-seconds 5400 \
  >> trainer.log 2>&1 &
NEW_PID=$!
echo "trainer pid $NEW_PID"

# No-strays verification: after the restart, every pid matching any service
# shape must be the new process (or a direct child of it). Anything else means
# a survivor slipped past the kill — print it loudly and exit non-zero. Do NOT
# auto-kill here: the operator must look before anything else dies.
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
  echo "FATAL: new $SERVICE (pid $NEW_PID) exited right after launch — check trainer.log"
  exit 1
fi
echo "verified: the only running $SERVICE is the new pid $NEW_PID (no strays)"
