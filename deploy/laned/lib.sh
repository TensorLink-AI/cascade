#!/usr/bin/env bash
# shellcheck shell=bash
# deploy/laned/lib.sh: shared code for the lane D S5 restart scripts
# (plan §5 row 10, §10.5 S5, §10.8 Rollback). Sourced by restart_*.sh.
#
# One run of a restart script, in order:
#   1. refusals: unknown refs, a release dir at the wrong commit, and the
#      quiet-window rules (window.py: freeze, clock, settlement, scoring,
#      in_flight, harvest; `--ack RULE` accepts one) — nothing is planned or
#      changed when any of them refuses (exit 1);
#   2. PLAN worktree  create (git worktree add --detach) or keep the release
#                     and the previous release (the rollback target);
#   3. PLAN check     <release>/scripts/deploy_gate.py check (R8);
#   4. PLAN snapshot  the §10.8 state files into <root>/_state_backups/<ts>/;
#   5. PLAN restart   the service on the release (python -P, PYTHONPATH,
#                     systemd scope, OOM score -500), or for the eval host an
#                     ssh sync of the release + `uv sync --frozen`;
#   6. PLAN verify    <release>/scripts/deploy_gate.py verify.
# --dry-run prints those lines and changes nothing. --execute prints each
# line as RUN and runs it, stopping at the first failure. A rollback is the
# same script with the release and previous-release flags swapped.
#
# Exit codes: 0 GO / done, 1 refused, 2 usage, 3 a step failed, 4 verify failed.

LANED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# §10.8 state files, all JSON/TOML text (validator_state.db too: the
# validator writes it with write_text_atomic). activation_state.json is the
# DEC-CA-0045 main tally, which the activation_*_state.json glob does not
# match; the .lane_b/.lane_c/.settlement sidecars sit next to the state db.
# Each file's <name>.bak (the loaders' fallback) is copied when present.
LANED_STATE_FIXED=(validator_state.db cascade_state.json warm_start_init.json
  lane_d_state.json netconfig_state.json activation_state.json
  satellite_state.json decided_votes.toml
  validator_state.db.lane_b.json validator_state.db.lane_c.json
  validator_state.db.settlement.json)

MODE=""
ROOT=""
REPO=""
VENV=""
RELEASE_REF=""
RELEASE_DIR=""
PREVIOUS_REF=""
PREVIOUS_DIR=""
HOST=""
PORT=""
KNOWN_HOSTS=""
REMOTE_DIR="/root/cascade"
ARGS_FILE=""
ACKS=()
SVC_ARGS=()
SVC=""
GATE_SVC=""
NOW=""
TS=""
PY=""
RELEASE_SHA=""
PREVIOUS_SHA=""
RELEASE_ACTION=""
PREVIOUS_ACTION=""
TAG="PLAN"

laned_usage() {
  local extra=" [--args-file FILE]"
  if [ "$SVC" = eval_host ]; then
    extra=" --host USER@HOST --port N --known-hosts FILE [--remote-dir DIR]"
  fi
  printf 'usage: %s (--dry-run|--execute) --root ROOT --repo REPO --venv VENV\n' "restart_$SVC.sh" >&2
  printf '         --release-ref REF --release-dir DIR --previous-ref REF --previous-dir DIR\n' >&2
  printf '        %s [--ack RULE ...]\n' "$extra" >&2
  if [ -n "${1:-}" ]; then
    printf 'error: %s\n' "$1" >&2
  fi
  exit 2
}

laned_refuse() {
  printf 'REFUSE [%s]: %s\n' "$1" "$2"
  exit 1
}

laned_die() {
  printf 'FAILED: %s\n' "$1" >&2
  exit "${2:-3}"
}

# Shell-quote argv for display, leaving plain words (paths, flags, k=v) as-is.
laned_quote() {
  local out="" a
  for a in "$@"; do
    case $a in
      '' | *[!A-Za-z0-9_@%+=:,./-]*) a="'${a//\'/\'\\\'\'}'" ;;
    esac
    out+="$a "
  done
  printf '%s' "${out% }"
}

laned_execute() { [ "$MODE" = execute ]; }

# One plan line whose text is exactly the argv it runs.
laned_step() {
  local step=$1
  shift
  printf '%s %s %s\n' "$TAG" "$step" "$(laned_quote "$@")"
  if laned_execute; then
    local rc=0
    "$@" || rc=$?
    [ "$rc" = 0 ] || laned_die "$step: $(laned_quote "$@") exited $rc"
  fi
}

# One plan line shown as TEXT, run by a function (pipes, redirections).
laned_step_shown() {
  local step=$1 text=$2
  shift 2
  printf '%s %s %s\n' "$TAG" "$step" "$text"
  if laned_execute; then
    local rc=0
    "$@" || rc=$?
    [ "$rc" = 0 ] || laned_die "$step: $text exited $rc"
  fi
}

laned_need_value() {
  [ "$#" -ge 2 ] && [ -n "$2" ] || laned_usage "$1 needs a value"
}

laned_abs_path() {
  case $2 in
    /*) ;;
    *) laned_usage "$1 must be an absolute path: $2" ;;
  esac
  case $2 in
    *[[:space:]]*) laned_usage "$1 must not contain whitespace: $2" ;;
  esac
}

laned_parse() {
  SVC=$1
  shift
  case $SVC in
    eval_host) GATE_SVC=eval-host ;;
    validator) GATE_SVC=validator ;;
    intake_laned) GATE_SVC=intake-laned ;;
    trainer) GATE_SVC=trainer ;;
    *) laned_die "unknown service $SVC" 2 ;;
  esac
  while [ "$#" -gt 0 ]; do
    case $1 in
      --dry-run | --execute)
        [ -z "$MODE" ] || laned_usage "give exactly one of --dry-run / --execute"
        MODE=${1#--}
        shift
        ;;
      --root | --repo | --venv | --release-ref | --release-dir | --previous-ref | --previous-dir | --ack)
        laned_need_value "$@"
        case $1 in
          --root) ROOT=$2 ;;
          --repo) REPO=$2 ;;
          --venv) VENV=$2 ;;
          --release-ref) RELEASE_REF=$2 ;;
          --release-dir) RELEASE_DIR=$2 ;;
          --previous-ref) PREVIOUS_REF=$2 ;;
          --previous-dir) PREVIOUS_DIR=$2 ;;
          --ack) ACKS+=("$2") ;;
        esac
        shift 2
        ;;
      --host | --port | --known-hosts | --remote-dir)
        [ "$SVC" = eval_host ] || laned_usage "$1 is an eval host flag"
        laned_need_value "$@"
        case $1 in
          --host) HOST=$2 ;;
          --port) PORT=$2 ;;
          --known-hosts) KNOWN_HOSTS=$2 ;;
          --remote-dir) REMOTE_DIR=$2 ;;
        esac
        shift 2
        ;;
      --args-file)
        [ "$SVC" != eval_host ] || laned_usage "--args-file is a box service flag"
        laned_need_value "$@"
        ARGS_FILE=$2
        shift 2
        ;;
      -h | --help) laned_usage ;;
      *) laned_usage "unknown argument: $1" ;;
    esac
  done
  [ -n "$MODE" ] || laned_usage "give --dry-run (print the plan) or --execute (run it)"
  local f
  for f in ROOT REPO VENV RELEASE_REF RELEASE_DIR PREVIOUS_REF PREVIOUS_DIR; do
    [ -n "${!f}" ] || laned_usage "missing --$(printf '%s' "$f" | tr 'A-Z_' 'a-z-')"
  done
  laned_abs_path --root "$ROOT"
  laned_abs_path --repo "$REPO"
  laned_abs_path --venv "$VENV"
  laned_abs_path --release-dir "$RELEASE_DIR"
  laned_abs_path --previous-dir "$PREVIOUS_DIR"
  [ -d "$ROOT" ] || laned_usage "--root $ROOT is not a directory"
  git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || laned_usage "--repo $REPO is not a git repo"
  PY="$VENV/bin/python"
  [ -x "$PY" ] || laned_usage "--venv $VENV has no executable bin/python"
  RELEASE_DIR="$(realpath -m -- "$RELEASE_DIR")"
  PREVIOUS_DIR="$(realpath -m -- "$PREVIOUS_DIR")"
  [ "$RELEASE_DIR" != "$PREVIOUS_DIR" ] || laned_usage "--release-dir and --previous-dir are the same"
  local ack
  for ack in "${ACKS[@]}"; do
    case $ack in
      *[!a-z_]*) laned_usage "--ack takes a rule name such as freeze, got: $ack" ;;
    esac
  done
  if [ "$SVC" = eval_host ]; then
    [ -n "$HOST" ] || laned_usage "missing --host"
    [ -n "$PORT" ] || laned_usage "missing --port"
    [ -n "$KNOWN_HOSTS" ] || laned_usage "missing --known-hosts"
    [[ $HOST =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$ ]] || laned_usage "--host must be USER@HOST: $HOST"
    [[ $PORT =~ ^[0-9]+$ ]] || laned_usage "--port must be a number: $PORT"
    laned_abs_path --known-hosts "$KNOWN_HOSTS"
    laned_abs_path --remote-dir "$REMOTE_DIR"
    [ -s "$KNOWN_HOSTS" ] || laned_usage "--known-hosts $KNOWN_HOSTS is missing or empty (pin the pod's host key first)"
  else
    [ -n "$ARGS_FILE" ] || ARGS_FILE="$ROOT/laned_args/$GATE_SVC.args"
    laned_abs_path --args-file "$ARGS_FILE"
  fi
  if laned_execute; then
    TAG="RUN"
  fi
}

# LANED_NOW (ISO 8601 UTC) replaces the clock in a dry-run; a real run uses
# the real clock only.
laned_clock() {
  if [ -n "${LANED_NOW:-}" ]; then
    laned_execute && laned_usage "LANED_NOW is a test clock: unset it for --execute"
    [[ $LANED_NOW =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] \
      || laned_usage "LANED_NOW must look like 2026-11-16T12:00:00Z, got: $LANED_NOW"
    NOW=$LANED_NOW
  else
    NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi
  TS="$(date -u -d "$NOW" +%Y%m%dT%H%M%SZ)" || laned_usage "unreadable time: $NOW"
}

laned_resolve_refs() {
  RELEASE_SHA="$(git -C "$REPO" rev-parse --verify --quiet "$RELEASE_REF^{commit}" || true)"
  [ -n "$RELEASE_SHA" ] || laned_refuse ref "--release-ref $RELEASE_REF is not a commit in $REPO"
  PREVIOUS_SHA="$(git -C "$REPO" rev-parse --verify --quiet "$PREVIOUS_REF^{commit}" || true)"
  [ -n "$PREVIOUS_SHA" ] || laned_refuse ref "--previous-ref $PREVIOUS_REF is not a commit in $REPO"
}

# WT_ACTION=create | keep, or refuse when DIR exists at another commit.
WT_ACTION=""
laned_worktree_action() {
  local dir=$1 ref=$2 sha=$3 top head
  if [ ! -e "$dir" ]; then
    WT_ACTION=create
    return 0
  fi
  top="$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null || true)"
  if [ -z "$top" ] || [ "$(realpath -m -- "$top")" != "$dir" ]; then
    laned_refuse worktree "$dir exists but is not a git worktree"
  fi
  head="$(git -C "$dir" rev-parse --verify --quiet HEAD || true)"
  if [ "$head" != "$sha" ]; then
    laned_refuse worktree "$dir is at ${head:-no commit}, not $ref ($sha); never reused at another commit"
  fi
  WT_ACTION=keep
}

laned_window() {
  local argv=("$PY" -P -B "$LANED_DIR/window.py" --root "$ROOT" --now "$NOW") ack rc=0
  for ack in "${ACKS[@]}"; do
    argv+=(--ack "$ack")
  done
  "${argv[@]}" || rc=$?
  case $rc in
    0) ;;
    1) exit 1 ;;
    *) laned_die "window check failed (exit $rc)" 2 ;;
  esac
}

laned_load_args() {
  [ "$SVC" = eval_host ] && return 0
  if [ ! -f "$ARGS_FILE" ]; then
    laned_execute && laned_refuse args "$ARGS_FILE is missing: the service's argument line (one argument per line) must be staged first"
    printf 'NOTE %s is missing: the plan below has no service arguments and --execute refuses [args]\n' "$ARGS_FILE"
    return 0
  fi
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in
      '' | '#'*) continue ;;
    esac
    SVC_ARGS+=("$line")
  done <"$ARGS_FILE"
}

laned_preflight() {
  laned_execute || return 0
  if [ "$SVC" != eval_host ]; then
    [ -r "$ROOT/.env" ] || laned_refuse env "$ROOT/.env is not readable: the service environment can't be loaded"
  fi
}

laned_emit_worktrees() {
  local label dir ref action
  for label in release previous; do
    if [ "$label" = release ]; then
      dir=$RELEASE_DIR ref=$RELEASE_REF action=$RELEASE_ACTION
    else
      dir=$PREVIOUS_DIR ref=$PREVIOUS_REF action=$PREVIOUS_ACTION
    fi
    if [ "$action" = keep ]; then
      printf '%s worktree keep %s %s at %s\n' "$TAG" "$label" "$dir" "$ref"
    else
      laned_step worktree git -C "$REPO" worktree add --detach "$dir" "$ref"
    fi
  done
}

laned_run_check() {
  local rc=0
  "$@" || rc=$?
  case $rc in
    0) return 0 ;;
    1) laned_refuse deploy_gate "deploy_gate check refused (see its REFUSE lines above)" ;;
    *) return "$rc" ;;
  esac
}

laned_emit_check() {
  local argv=("$PY" -P "$RELEASE_DIR/scripts/deploy_gate.py" check --service "$GATE_SVC"
    --from "$PREVIOUS_REF" --to "$RELEASE_REF" --repo "$REPO" --root "$ROOT") ack
  for ack in "${ACKS[@]}"; do
    argv+=(--ack "$ack")
  done
  laned_step_shown check "$(laned_quote "${argv[@]}")" laned_run_check "${argv[@]}"
}

# Copy <path>.bak (the previous good file) into BACKUP when it exists.
laned_snapshot_bak() {
  local bak="$1.bak"
  if [ -e "$bak" ]; then
    laned_step snapshot cp -a "$bak" "$2/${bak##*/}"
  fi
}

laned_emit_snapshot() {
  local backup="$ROOT/_state_backups/$TS" name path found=0
  laned_step snapshot mkdir -p "$ROOT/_state_backups"
  laned_step snapshot mkdir "$backup"
  for name in "${LANED_STATE_FIXED[@]}"; do
    path="$ROOT/$name"
    if [ ! -e "$path" ]; then
      printf '%s snapshot %s absent (nothing to copy)\n' "$TAG" "$name"
    else
      laned_step snapshot cp -a "$path" "$backup/$name"
    fi
    laned_snapshot_bak "$path" "$backup"
  done
  for path in "$ROOT"/activation_*_state.json; do
    [ -e "$path" ] || continue
    found=1
    laned_step snapshot cp -a "$path" "$backup/${path##*/}"
    laned_snapshot_bak "$path" "$backup"
  done
  if [ "$found" = 0 ]; then
    printf '%s snapshot activation_*_state.json absent (nothing to copy)\n' "$TAG"
  fi
}

# Processes running SVC: the venv's python on the console script, or `python -m
# <package>`, under any path prefix (never a shell that merely names them).
laned_proc_patterns() {
  case $SVC in
    validator)
      printf '%s\n' '^[^ ]*python[0-9.]*( -[A-Za-z]+)* [^ ]*bin/cascade-validator( |$)' \
        '^[^ ]*python[0-9.]*( -[A-Za-z]+)* -m cascade\.validator(\.main)?( |$)'
      ;;
    trainer)
      printf '%s\n' '^[^ ]*python[0-9.]*( -[A-Za-z]+)* [^ ]*bin/cascade-trainer( |$)' \
        '^[^ ]*python[0-9.]*( -[A-Za-z]+)* -m cascade\.trainer(\.main)?( |$)'
      ;;
    intake_laned)
      printf '%s\n' '^[^ ]*python[0-9.]*( -[A-Za-z]+)* [^ ]*bin/cascade-laned-intake( |$)' \
        '^[^ ]*python[0-9.]*( -[A-Za-z]+)* -m cascade\.lane_d(\.[a-z_]+)?( |$)'
      ;;
  esac
}

laned_pids() {
  local pat
  while IFS= read -r pat; do
    pgrep -f -- "$pat" || true
  done <<<"$(laned_proc_patterns)" | sort -un | grep -v -x -e "$$" -e "$PPID" || true
}

# TERM, KILL after 30 s, and fail (no launch) if any process survives.
laned_stop() {
  local pids i
  pids="$(laned_pids)"
  if [ -z "$pids" ]; then
    printf 'no running %s process\n' "$GATE_SVC"
    return 0
  fi
  local arr=()
  mapfile -t arr <<<"$pids"
  ps -o pid,lstart,cmd -p "$(IFS=,; printf '%s' "${arr[*]}")" || true
  kill -TERM "${arr[@]}" 2>/dev/null || true
  for i in $(seq 1 30); do
    sleep 1
    pids="$(laned_pids)"
    if [ -z "$pids" ]; then
      printf 'stopped after %s s\n' "$i"
      return 0
    fi
  done
  mapfile -t arr <<<"$pids"
  printf 'TERM did not stick after 30 s: KILL %s\n' "${arr[*]}"
  kill -KILL "${arr[@]}" 2>/dev/null || true
  sleep 2
  pids="$(laned_pids)"
  if [ -n "$pids" ]; then
    printf 'old %s process(es) survived TERM+KILL, not launching: %s\n' "$GATE_SVC" "$pids" >&2
    return 1
  fi
}

laned_launch() {
  local out=$1
  shift
  (
    cd "$ROOT" || exit 1
    set -a
    # shellcheck source=/dev/null
    . "$ROOT/.env"
    set +a
    nohup "$@" >>"$out" 2>&1 &
    printf 'launched pid %s\n' "$!"
  )
}

laned_emit_restart_box() {
  local entry out
  case $SVC in
    validator) entry=("$VENV/bin/cascade-validator") out="$ROOT/validator.log" ;;
    trainer) entry=("$VENV/bin/cascade-trainer") out="$ROOT/trainer.out" ;;
    intake_laned) entry=(-m cascade.lane_d) out="$ROOT/intake-laned.log" ;;
  esac
  local unit="cascade-$GATE_SVC-$TS"
  local argv=(systemd-run --scope "--unit=$unit" "--description=cascade $GATE_SVC $RELEASE_REF"
    choom -n -500 -- env "PYTHONPATH=$RELEASE_DIR" "$PY" -P "${entry[@]}" "${SVC_ARGS[@]}")
  laned_step_shown restart \
    "stop every running $GATE_SVC process (console script or module form): TERM, KILL after 30 s, no launch if one survives" \
    laned_stop
  laned_step_shown restart \
    "cd $ROOT && set -a && . $ROOT/.env && set +a && nohup $(laned_quote "${argv[@]}") >> $out 2>&1 &" \
    laned_launch "$out" "${argv[@]}"
}

laned_sync_eval_host() {
  local -a rsync_argv=("$@")
  git -C "$RELEASE_DIR" ls-files -z | "${rsync_argv[@]}"
}

laned_emit_restart_eval_host() {
  local ssh_e="ssh -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=yes -o BatchMode=yes -p $PORT"
  local ssh_argv=(ssh -o "UserKnownHostsFile=$KNOWN_HOSTS" -o StrictHostKeyChecking=yes
    -o BatchMode=yes -p "$PORT" "$HOST")
  local rsync_argv=(rsync -a --from0 --files-from=- -e "$ssh_e" "$RELEASE_DIR/" "$HOST:$REMOTE_DIR/")
  laned_step restart "${ssh_argv[@]}" "mkdir -p $REMOTE_DIR"
  laned_step_shown restart \
    "git -C $RELEASE_DIR ls-files -z | $(laned_quote "${rsync_argv[@]}")" \
    laned_sync_eval_host "${rsync_argv[@]}"
  laned_step restart "${ssh_argv[@]}" "cd $REMOTE_DIR && PATH=\"\$HOME/.local/bin:\$PATH\" uv sync --frozen"
  laned_step restart "${ssh_argv[@]}" \
    "cd $REMOTE_DIR && .venv/bin/python -P -c 'import cascade.validator.private_worker'"
}

laned_run_verify() {
  local attempt rc=1
  for attempt in 1 2 3 4 5 6; do
    sleep 20
    rc=0
    "$@" || rc=$?
    [ "$rc" = 0 ] && return 0
    printf 'verify attempt %s/6 not yet verified\n' "$attempt"
  done
  laned_die "deploy_gate verify failed for $GATE_SVC: roll back with the same script, --release-ref $PREVIOUS_REF --release-dir $PREVIOUS_DIR --previous-ref $RELEASE_REF --previous-dir $RELEASE_DIR" 4
}

laned_emit_verify() {
  local argv=("$PY" -P "$RELEASE_DIR/scripts/deploy_gate.py" verify --service "$GATE_SVC"
    --release "$RELEASE_DIR" --root "$ROOT")
  laned_step_shown verify "$(laned_quote "${argv[@]}")" laned_run_verify "${argv[@]}"
}

laned_main() {
  laned_parse "$@"
  laned_clock
  laned_resolve_refs
  laned_worktree_action "$RELEASE_DIR" "$RELEASE_REF" "$RELEASE_SHA"
  RELEASE_ACTION=$WT_ACTION
  laned_worktree_action "$PREVIOUS_DIR" "$PREVIOUS_REF" "$PREVIOUS_SHA"
  PREVIOUS_ACTION=$WT_ACTION
  laned_window
  laned_load_args
  laned_preflight
  printf 'GO %s %s -> %s (%s mode, clock %s)\n' "$GATE_SVC" "$PREVIOUS_REF" "$RELEASE_REF" "$MODE" "$NOW"
  laned_emit_worktrees
  laned_emit_check
  laned_emit_snapshot
  if [ "$SVC" = eval_host ]; then
    laned_emit_restart_eval_host
  else
    laned_emit_restart_box
  fi
  laned_emit_verify
  if laned_execute; then
    printf 'DONE %s on %s\n' "$GATE_SVC" "$RELEASE_REF"
  fi
}
