#!/bin/sh
# Keep the gauntlet stack on the newest published images, without ever killing
# a job in flight.
#
#   updater.sh            loop forever (the compose `updater` service)
#   updater.sh --once     one check-and-update pass (by hand / host cron)
#
# Handshake with the judge (cascade/miner/harness/gauntlet.py):
#   1. pull; if any service's image changed, touch <workdir>/RESTART
#   2. the judge finishes its current cycle, writes RESTART.ack and exits;
#      restarted by its policy, it stays PARKED while RESTART + ack exist
#   3. recreate every service on the new images, then clear RESTART + ack;
#      the new judge un-parks and resumes from disk
# Workers and the operator hold no in-flight state while the judge is parked.
set -eu

: "${STACK_DIR:?set STACK_DIR (the deploy/harness directory, same path as on the host)}"
WD="${GAUNTLET_DIR:-$STACK_DIR/work/gauntlet}"
INTERVAL="${UPDATE_INTERVAL_SECONDS:-21600}"
ACK_TIMEOUT="${ACK_TIMEOUT_SECONDS:-172800}"
SERVICES="judge worker operator"

dc() { docker compose --project-directory "$STACK_DIR" -f "$STACK_DIR/docker-compose.yml" "$@"; }
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) updater: $*"; }

changed_services() {
  out=""
  for s in $SERVICES; do
    cid="$(dc ps -q "$s" | head -n 1)"
    [ -n "$cid" ] || continue
    running="$(docker inspect -f '{{.Image}}' "$cid")"
    ref="$(docker inspect -f '{{.Config.Image}}' "$cid")"
    latest="$(docker image inspect -f '{{.Id}}' "$ref" 2>/dev/null || true)"
    if [ -n "$latest" ] && [ "$running" != "$latest" ]; then out="$out $s"; fi
  done
  echo "$out"
}

judge_running() { [ -n "$(dc ps -q --status running judge)" ]; }

update_once() {
  if ! dc pull -q $SERVICES; then
    log "pull failed; keeping the current images"
    return 0
  fi
  changed="$(changed_services)"
  if [ -z "$changed" ]; then
    log "up to date"
    return 0
  fi
  log "new images for:$changed"
  mkdir -p "$WD"
  if judge_running; then
    touch "$WD/RESTART"
    log "asked the judge to park after its current cycle"
    waited=0
    until [ -f "$WD/RESTART.ack" ]; do
      if [ "$waited" -ge "$ACK_TIMEOUT" ]; then
        log "judge did not park within ${ACK_TIMEOUT}s; leaving RESTART in place"
        return 1
      fi
      sleep 30
      waited=$((waited + 30))
    done
  fi
  dc up -d --no-build $SERVICES
  rm -f "$WD/RESTART" "$WD/RESTART.ack"
  log "recreated:$changed"
}

if [ "${1:-}" = "--once" ]; then
  update_once
  exit $?
fi
while true; do
  update_once || true
  sleep "$INTERVAL"
done
