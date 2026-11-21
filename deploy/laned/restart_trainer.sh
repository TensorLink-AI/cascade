#!/usr/bin/env bash
# Restart the trainer onto a release worktree (lane D S5, plan §10.5; last in
# the S5 order, never with legs in flight; rollback §10.8: the same script with
# the release/previous flags swapped). Always run --dry-run first. See lib.sh.
#
#   restart_trainer.sh (--dry-run|--execute) --root ROOT --repo REPO --venv VENV \
#       --release-ref REF --release-dir DIR --previous-ref REF --previous-dir DIR \
#       [--args-file FILE] [--ack RULE ...]
set -euo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
laned_main trainer "$@"
