#!/usr/bin/env bash
# Restart the lane D intake (python -P -m cascade.lane_d, in a systemd scope)
# onto a release worktree (lane D S5, plan §10.5; rollback §10.8: the same
# script with the release/previous flags swapped). The lane A intake
# (cascade-intake.service, cascade.funding.main) is not touched. Always run
# --dry-run first. See lib.sh.
#
#   restart_intake_laned.sh (--dry-run|--execute) --root ROOT --repo REPO --venv VENV \
#       --release-ref REF --release-dir DIR --previous-ref REF --previous-dir DIR \
#       [--args-file FILE] [--ack RULE ...]
set -euo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
laned_main intake_laned "$@"
