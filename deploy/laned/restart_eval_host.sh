#!/usr/bin/env bash
# Sync the private eval host (a rented pod; the validator reaches it over ssh
# per eval) to a release: copy the release's tracked files to --remote-dir,
# `uv sync --frozen --inexact`, import-probe the worker. Every ssh pins the pod's host
# key (--known-hosts, StrictHostKeyChecking=yes). First in the S5 order
# (plan §10.5); rollback §10.8: the same script with the release/previous
# flags swapped. Always run --dry-run first. See lib.sh.
#
#   restart_eval_host.sh (--dry-run|--execute) --root ROOT --repo REPO --venv VENV \
#       --release-ref REF --release-dir DIR --previous-ref REF --previous-dir DIR \
#       --host USER@HOST --port N --known-hosts FILE [--remote-dir DIR] [--ack RULE ...]
set -euo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
laned_main eval_host "$@"
