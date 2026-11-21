#!/usr/bin/env bash
# Restart the main validator onto a release worktree (lane D S5, plan §10.5;
# rollback §10.8: the same script with the release/previous flags swapped).
# Always run --dry-run first and paste the plan to the owner. See lib.sh.
#
#   restart_validator.sh (--dry-run|--execute) --root ROOT --repo REPO --venv VENV \
#       --release-ref REF --release-dir DIR --previous-ref REF --previous-dir DIR \
#       [--args-file FILE] [--ack RULE ...]
set -euo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
laned_main validator "$@"
