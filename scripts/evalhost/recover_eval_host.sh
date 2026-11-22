#!/usr/bin/env bash
# R12 owner-paste recovery of an eval host after a fault (LANE_D_PLAN §10.4,
# §10.5 S4; OPSLOG 2026-11-22: Lium's NAT moved a RUNNING pod's IP with its
# host key unchanged). Prints the plan (--dry-run) or a reviewed, pasteable
# block (--print); it never runs a remote command itself. Steps, in order:
#   resolve_endpoint  lium ps, then evalhost.py resolve-endpoint: the pod is
#                     only called gone on not_found, never before this step
#   known_hosts       pin-host-key (re-pin under the new IP only when the
#                     scanned key equals the key pinned for the old one)
#   on_rent           nvidia-smi -L, the GPU query, df (two-checkpoint headroom)
#   venv_sync         evalhost.py sync (release files, then uv sync --frozen)
#   handshake         suite identities (bench profiles) + the W4 worker hello
#   evidence          evalhost.py evidence -> evidence/r12_pod.json
# Free disk is reported, never freed here; no pod is stopped or removed.
set -euo pipefail

usage() {
    cat <<'EOF'
usage: recover_eval_host.sh --profile P --pod NAME --configured HOST:PORT --user USER \
    --known-hosts F --release-dir D --work-dir W [--dest DIR] (--dry-run | --print)

  --dry-run     print `PLAN <step> <command>` lines; runs nothing, writes nothing
  --print       print a self-contained bash block (set -euo pipefail, every value
                literal, a `# STEP <step>` line per step) to review and paste
  --configured  the endpoint the pod is configured (and pinned) under
  --dest        the release directory on the host (default /root/cascade)

The printed block stops at the first verdict that needs a human: a changed
host key, a pod that is not RUNNING, or a pod lium ps no longer lists.
EOF
}

# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

EH_TITLE="eval-host recovery"
configured=""
user=""
while (($#)); do
    case $1 in
        -h | --help) usage; exit 0 ;;
        --dry-run) EH_MODE=dry-run; shift ;;
        --print) EH_MODE=print; shift ;;
        --profile | --pod | --configured | --user | --known-hosts | --release-dir | --work-dir | --dest)
            (($# >= 2)) || eh_die "$1 needs a value"
            case $1 in
                --profile) EH_PROFILE=$2 ;;
                --pod) EH_POD=$2 ;;
                --configured) configured=$2 ;;
                --user) user=$2 ;;
                --known-hosts) EH_KNOWN_HOSTS=$2 ;;
                --release-dir) EH_RELEASE=$2 ;;
                --work-dir) EH_WORK=$2 ;;
                --dest) EH_DEST=$2 ;;
            esac
            shift 2 ;;
        *) usage >&2; eh_die "unknown argument: $1" ;;
    esac
done

[[ -n $configured ]] || eh_die "missing --configured HOST:PORT"
[[ -n $user ]] || eh_die "missing --user"
eh_check_common
[[ $configured =~ ^[A-Za-z0-9._-]+:[1-9][0-9]{0,4}$ ]] || eh_die "--configured must be HOST:PORT"
[[ $user =~ ^[A-Za-z0-9._-]+$ ]] || eh_die "--user must be a login name"

tool=$(eh_tool)
work=$(eh_q "$EH_WORK")
kh=$(eh_q "$EH_KNOWN_HOSTS")
scan=$(eh_q "$EH_WORK/ssh_keyscan.txt")
resolve="$tool resolve-endpoint --lium-ps $(eh_q "$EH_WORK/lium_ps.json") --pod $(eh_q "$EH_POD") --configured $configured"

eh_add resolve_endpoint "mkdir -p $work"
eh_add resolve_endpoint "lium ps --format json > $(eh_q "$EH_WORK/lium_ps.json")"
eh_add resolve_endpoint "rc=0; $resolve --endpoint-out $(eh_q "$EH_WORK/endpoint.txt") > $(eh_q "$EH_WORK/resolve.json") || rc=\$?"
eh_add resolve_endpoint "cat $(eh_q "$EH_WORK/resolve.json")"
eh_add resolve_endpoint "case \"\$rc\" in 0|34) ;; *) echo \"resolve-endpoint exit \$rc: stop (32 not_found is the only verdict that calls the pod gone)\" >&2; exit \"\$rc\" ;; esac"
eh_add resolve_endpoint "read -r EP_HOST EP_PORT < $(eh_q "$EH_WORK/endpoint.txt")"
eh_add resolve_endpoint "ssh-keyscan -p \"\$EP_PORT\" \"\$EP_HOST\" > $scan"
eh_add resolve_endpoint "rc=0; $resolve --known-hosts $kh --scanned $scan > $(eh_q "$EH_WORK/resolve_verified.json") || rc=\$?"
eh_add resolve_endpoint "cat $(eh_q "$EH_WORK/resolve_verified.json")"
eh_add resolve_endpoint "case \"\$rc\" in 0|30) ;; *) echo \"resolve-endpoint exit \$rc with the scanned key: stop (31 = a different host key)\" >&2; exit \"\$rc\" ;; esac"

pin="$tool pin-host-key --known-hosts $kh --host \"\$EP_HOST\" --port \"\$EP_PORT\" --scanned $scan"
eh_add known_hosts "if [ \"\$EP_HOST:\$EP_PORT\" = $configured ]; then $pin; else $pin --previous $configured; fi"

eh_common_steps "\"\$EP_HOST\"" "\"\$EP_PORT\"" "$user"

eh_emit "$EH_MODE"
