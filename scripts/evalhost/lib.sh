# shellcheck shell=bash
# Shared plan builder for setup_eval_host.sh and recover_eval_host.sh (R12).
# Sourced, never run. Builds an ordered list of (step, command line) pairs and
# prints it as `PLAN <step> <command>` lines (--dry-run) or as a pasteable
# bash block with a `# STEP <step>` header per step (--print). Nothing here
# runs a remote command, writes a file, or reads a secret: the only process it
# starts is `evalhost.py profile-info`, which reads the profile.

EH_KIT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
EH_REPO_ROOT=$(cd -- "$EH_KIT_DIR/../.." && pwd -P)
EH_TOOL="$EH_KIT_DIR/evalhost.py"
EH_PY=${EVALHOST_PYTHON:-python3}

EH_STEPS=()
EH_CMDS=()

# Filled by the calling script's flag parser.
EH_TITLE=""
EH_MODE=""
EH_PROFILE=""
EH_POD=""
EH_KNOWN_HOSTS=""
EH_RELEASE=""
EH_WORK=""
EH_DEST="/root/cascade"

eh_die() {
    printf '%s: %s\n' "${0##*/}" "$1" >&2
    exit 2
}

# Shell-quote one word (single quotes; plain words stay bare).
eh_q() {
    local s=$1
    if [[ $s =~ ^[A-Za-z0-9_./@:,+=%-]+$ ]]; then
        printf '%s' "$s"
    else
        s=${s//\'/\'\\\'\'}
        printf "'%s'" "$s"
    fi
}

# An absolute path for a value that may not exist yet (no symlink resolution).
eh_abs() {
    realpath -m -s -- "$1"
}

eh_add() {
    EH_STEPS+=("$1")
    EH_CMDS+=("$2")
}

# The evalhost.py command prefix, always `python -P` (cwd never shadows the kit).
eh_tool() {
    printf '%s -P %s' "$(eh_q "$EH_PY")" "$(eh_q "$EH_TOOL")"
}

# One profile value, read through the kit (validates the profile too).
eh_profile() {
    "$EH_PY" -P "$EH_TOOL" profile-info --profile "$1" --get "$2"
}

# A pinned ssh prefix: eh_ssh PORT_WORD USERHOST_WORD (both already shell words).
# -n: ssh never reads the block's own stdin.
eh_ssh() {
    printf 'ssh -n -p %s -o UserKnownHostsFile=%s -o StrictHostKeyChecking=yes -o BatchMode=yes %s' \
        "$1" "$(eh_q "$EH_KNOWN_HOSTS")" "$2"
}

# Capture a remote command's stdout into the work dir:
# eh_capture STEP PORT_WORD USERHOST_WORD REMOTE_CMD FILE
eh_capture() {
    local remote
    remote=$(eh_q "$4")
    eh_add "$1" "$(eh_ssh "$2" "$3") $remote > $(eh_q "$EH_WORK/$5")"
}

# The on_rent / venv_sync / handshake / evidence steps shared by both scripts.
# eh_common_steps HOST_WORD PORT_WORD USER (HOST/PORT may be "$VAR" words).
eh_common_steps() {
    local host=$1 port=$2 user=$3 tool userhost bench_flag="" ids_flag=""
    tool=$(eh_tool)
    userhost="$(eh_q "$user")@$host"
    local smi_list smi_query df_cmd pins
    smi_list=$(eh_profile "$EH_PROFILE" smi_list_cmd)
    smi_query=$(eh_profile "$EH_PROFILE" smi_query_cmd)
    df_cmd=$(eh_profile "$EH_PROFILE" df_cmd)
    pins=$(eh_profile "$EH_PROFILE" suite_pins)

    eh_capture on_rent "$port" "$userhost" "$smi_list" nvidia_smi_L.txt
    eh_capture on_rent "$port" "$userhost" "$smi_query" nvidia_smi_query.txt
    eh_capture on_rent "$port" "$userhost" "$df_cmd" df.txt
    eh_add on_rent "$tool on-rent-check --profile $(eh_q "$EH_PROFILE") --nvidia-smi-L $(eh_q "$EH_WORK/nvidia_smi_L.txt") --nvidia-smi-query $(eh_q "$EH_WORK/nvidia_smi_query.txt") --df $(eh_q "$EH_WORK/df.txt")"

    if [[ $pins == 1 ]]; then
        bench_flag=" --bench-venv"
    fi
    # The profile's optional-dependencies groups (launch_eval: train, the
    # pinned torch): `uv sync --frozen --extra NAME` on the host (#784).
    local extras extra_flags="" e
    local -a extra_list=()
    extras=$(eh_profile "$EH_PROFILE" uv_extras)
    read -r -a extra_list <<< "$extras"
    for e in "${extra_list[@]}"; do
        extra_flags+=" --extra $(eh_q "$e")"
    done
    eh_add venv_sync "$tool sync --release-dir $(eh_q "$EH_RELEASE") --host $userhost --port $port --known-hosts $(eh_q "$EH_KNOWN_HOSTS") --dest $(eh_q "$EH_DEST")$bench_flag$extra_flags"

    local in_dest
    in_dest="cd $(eh_q "$EH_DEST") &&"
    if [[ $pins == 1 ]]; then
        eh_capture handshake "$port" "$userhost" "$in_dest $(eh_profile "$EH_PROFILE" bench_identity_cmd)" bench_identity.json
        eh_capture handshake "$port" "$userhost" "$(eh_profile "$EH_PROFILE" bench_files_cmd)" bench_files.txt
        eh_add handshake "$tool identities --bench-identity $(eh_q "$EH_WORK/bench_identity.json") --bench-files $(eh_q "$EH_WORK/bench_files.txt") --out $(eh_q "$EH_WORK/identities.json")"
        ids_flag=" --identities $(eh_q "$EH_WORK/identities.json")"
    fi
    eh_capture handshake "$port" "$userhost" "$in_dest $(eh_profile "$EH_PROFILE" hello_cmd)" hello.json
    eh_add handshake "$tool handshake --profile $(eh_q "$EH_PROFILE")$ids_flag --hello $(eh_q "$EH_WORK/hello.json") --release-dir $(eh_q "$EH_RELEASE")"

    eh_add evidence "$tool evidence --profile $(eh_q "$EH_PROFILE") --pod $(eh_q "$EH_POD") --host $host --port $port --captures $(eh_q "$EH_WORK") --release-dir $(eh_q "$EH_RELEASE") --out $(eh_q "$EH_REPO_ROOT/evidence/r12_pod.json")"
}

# eh_emit dry-run|print
eh_emit() {
    local mode=$1 i last=""
    if [[ $mode == print ]]; then
        printf '%s\n' "# ${EH_TITLE} (R12), generated by ${0##*/}; every value below is literal."
        printf '%s\n' "# Review it, save it to a file on the owner box and run it with bash: it"
        printf '%s\n' "# stops at the first failing step (set -e), which ends that bash only."
        printf '%s\n' "set -euo pipefail"
    fi
    for i in "${!EH_STEPS[@]}"; do
        if [[ $mode == dry-run ]]; then
            printf 'PLAN %s %s\n' "${EH_STEPS[i]}" "${EH_CMDS[i]}"
        else
            if [[ ${EH_STEPS[i]} != "$last" ]]; then
                printf '\n# STEP %s\n' "${EH_STEPS[i]}"
                last=${EH_STEPS[i]}
            fi
            printf '%s\n' "${EH_CMDS[i]}"
        fi
    done
}

# Validate the values every plan needs; make paths absolute.
eh_check_common() {
    [[ -n ${EH_PROFILE:-} ]] || eh_die "missing --profile"
    [[ -n ${EH_POD:-} ]] || eh_die "missing --pod"
    [[ -n ${EH_KNOWN_HOSTS:-} ]] || eh_die "missing --known-hosts"
    [[ -n ${EH_RELEASE:-} ]] || eh_die "missing --release-dir"
    [[ -n ${EH_WORK:-} ]] || eh_die "missing --work-dir"
    [[ -n ${EH_MODE:-} ]] || eh_die "pick --dry-run (plan) or --print (pasteable block)"
    [[ -f $EH_PROFILE ]] || eh_die "profile $EH_PROFILE does not exist"
    [[ $EH_POD =~ ^[A-Za-z0-9._-]+$ ]] || eh_die "--pod must be a pod name or huid"
    EH_PROFILE=$(eh_abs "$EH_PROFILE")
    EH_KNOWN_HOSTS=$(eh_abs "$EH_KNOWN_HOSTS")
    EH_RELEASE=$(eh_abs "$EH_RELEASE")
    EH_WORK=$(eh_abs "$EH_WORK")
    [[ $EH_KNOWN_HOSTS =~ ^[A-Za-z0-9_./@:+,=-]+$ ]] \
        || eh_die "--known-hosts must be a plain path (no spaces or quotes)"
    [[ $EH_DEST =~ ^/[A-Za-z0-9_./@:+,=-]*$ ]] || eh_die "--dest must be a plain absolute path"
    eh_profile "$EH_PROFILE" name > /dev/null || eh_die "profile $EH_PROFILE is not a valid profile"
}
