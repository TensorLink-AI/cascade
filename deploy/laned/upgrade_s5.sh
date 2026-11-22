#!/usr/bin/env bash
# Lane D S5 upgrade (plan §5 row 10, §10.5): the four restart scripts in the S5
# order, eval host -> validator -> lane D intake -> trainer, with the same
# flags. Stops at the first script that refuses or fails; later services are
# left untouched. Always run --dry-run first and paste the plan to the owner.
#
#   upgrade_s5.sh (--dry-run|--execute) --root ROOT --repo REPO --venv VENV \
#       --release-ref REF --release-dir DIR --previous-ref REF --previous-dir DIR \
#       --host USER@HOST --port N --known-hosts FILE [--remote-dir DIR] [--ack RULE ...]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
common=()
eval_only=()
while [ "$#" -gt 0 ]; do
  case $1 in
    --dry-run | --execute)
      common+=("$1")
      shift
      ;;
    --root | --repo | --venv | --release-ref | --release-dir | --previous-ref | --previous-dir | --ack)
      if [ "$#" -lt 2 ]; then
        printf 'upgrade_s5.sh: %s needs a value\n' "$1" >&2
        exit 2
      fi
      common+=("$1" "$2")
      shift 2
      ;;
    --host | --port | --known-hosts | --remote-dir)
      if [ "$#" -lt 2 ]; then
        printf 'upgrade_s5.sh: %s needs a value\n' "$1" >&2
        exit 2
      fi
      eval_only+=("$1" "$2")
      shift 2
      ;;
    *)
      printf 'upgrade_s5.sh: unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

for svc in eval_host validator intake_laned trainer; do
  printf '== S5 %s\n' "$svc"
  argv=("${common[@]}")
  if [ "$svc" = eval_host ]; then
    argv+=("${eval_only[@]}")
  fi
  rc=0
  bash "$here/restart_$svc.sh" "${argv[@]}" || rc=$?
  if [ "$rc" != 0 ]; then
    printf 'S5 stopped at %s (exit %s); the services after it were not touched\n' "$svc" "$rc"
    exit "$rc"
  fi
done
printf 'S5 complete: eval host, validator, lane D intake, trainer\n'
