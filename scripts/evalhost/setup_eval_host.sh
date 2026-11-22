#!/usr/bin/env bash
# R12 owner-paste bring-up of an eval host (LANE_D_PLAN §10.2, §10.4).
# Prints the plan (--dry-run) or a reviewed, pasteable block (--print); it
# never runs a remote command itself. Steps, in order:
#   known_hosts  ssh-keyscan into the work dir, then evalhost.py pin-host-key
#                (the key is pinned before any ssh)
#   on_rent      nvidia-smi -L, the GPU query, df, then evalhost.py on-rent-check
#   venv_sync    evalhost.py sync (release files, then uv sync --frozen)
#   handshake    suite identities (bench profiles) + the W4 worker hello
#   evidence     evalhost.py evidence -> evidence/r12_pod.json
set -euo pipefail

usage() {
    cat <<'EOF'
usage: setup_eval_host.sh --profile P --pod NAME --host USER@HOST --port N \
    --known-hosts F --release-dir D --work-dir W [--dest DIR] (--dry-run | --print)

  --dry-run   print `PLAN <step> <command>` lines; runs nothing, writes nothing
  --print     print a self-contained bash block (set -euo pipefail, every value
              literal, a `# STEP <step>` line per step) to review and paste
  --dest      the release directory on the host (default /root/cascade); for the
              code handshake it must be a git checkout of the release's HEAD

Every ssh line pins the host key (UserKnownHostsFile=<--known-hosts>,
StrictHostKeyChecking=yes); keys are only ever pinned via evalhost.py
pin-host-key. Nothing is deleted, and no pod is stopped.
EOF
}

# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

EH_TITLE="eval-host setup"
user_host=""
port=""
while (($#)); do
    case $1 in
        -h | --help) usage; exit 0 ;;
        --dry-run) EH_MODE=dry-run; shift ;;
        --print) EH_MODE=print; shift ;;
        --profile | --pod | --host | --port | --known-hosts | --release-dir | --work-dir | --dest)
            (($# >= 2)) || eh_die "$1 needs a value"
            case $1 in
                --profile) EH_PROFILE=$2 ;;
                --pod) EH_POD=$2 ;;
                --host) user_host=$2 ;;
                --port) port=$2 ;;
                --known-hosts) EH_KNOWN_HOSTS=$2 ;;
                --release-dir) EH_RELEASE=$2 ;;
                --work-dir) EH_WORK=$2 ;;
                --dest) EH_DEST=$2 ;;
            esac
            shift 2 ;;
        *) usage >&2; eh_die "unknown argument: $1" ;;
    esac
done

[[ -n $user_host ]] || eh_die "missing --host USER@HOST"
[[ -n $port ]] || eh_die "missing --port"
eh_check_common
[[ $user_host =~ ^([A-Za-z0-9._-]+)@([A-Za-z0-9.:_-]+)$ ]] || eh_die "--host must be USER@HOST"
user=${BASH_REMATCH[1]}
host=${BASH_REMATCH[2]}
if ! [[ $port =~ ^[1-9][0-9]{0,4}$ ]] || ((port < 1 || port > 65535)); then
    eh_die "--port must be 1..65535"
fi

tool=$(eh_tool)
work=$(eh_q "$EH_WORK")
kh=$(eh_q "$EH_KNOWN_HOSTS")

eh_add known_hosts "mkdir -p $work"
eh_add known_hosts "ssh-keyscan -p $port $host > $(eh_q "$EH_WORK/ssh_keyscan.txt")"
eh_add known_hosts "$tool pin-host-key --known-hosts $kh --host $host --port $port --scanned $(eh_q "$EH_WORK/ssh_keyscan.txt")"
eh_common_steps "$host" "$port" "$user"

eh_emit "$EH_MODE"
