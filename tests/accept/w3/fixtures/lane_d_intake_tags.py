"""The W3 intake's log event tags (plan §3.1 build rule 5).

``follower_patterns.py --patterns`` reads the string literals below: every
lane D log call under ``cascade/lane_d/`` must start with one of these tags.
cascade-ops (W11) adds the same tags to ``cascade_ops/events.py``.
"""

LANE_D_INTAKE_TAGS = (
    "lane_d_intake_up",            # startup: carries 'intake-laned up:' (deploy_gate verify)
    "lane_d_intake_submit",        # a granted submit
    "lane_d_intake_ready",         # a recorded ready
    "lane_d_intake_refused",       # any typed 4xx refusal (code only, never a grant)
    "lane_d_intake_expired",       # an entry moved to ready_expired (B10)
    "lane_d_intake_closed",        # an entry closed
    "lane_d_intake_unavailable",   # any typed 503 (storage / chain / grants / state)
    "lane_d_coldkey_unresolved",   # plan §1.2 D22 alert name
)

# The lane D recurring jobs (cascade/lane_d/ops_jobs.py, ops_jobs.LOG_TAGS; the
# docs/OPERATOR.md "Lane D recurring jobs" watcher table lists the same set).
LANE_D_OPS_TAGS = (
    "lane_d_ops_house_current",
    "lane_d_ops_house_no_lane_a_king",
    "lane_d_ops_house_resign_planned",
    "lane_d_ops_house_resigned",
    "lane_d_ops_house_resign_blocked",
    "lane_d_ops_house_failed",
    "lane_d_ops_evidence_current",
    "lane_d_ops_evidence_planned",
    "lane_d_ops_evidence_installed",
    "lane_d_ops_evidence_refused",
    "lane_d_ops_ledger_current",
    "lane_d_ops_ledger_planned",
    "lane_d_ops_ledger_written",
    "lane_d_ops_ledger_failed",
    "lane_d_ops_king_ok",
    "lane_d_ops_king_unpublished",
    "lane_d_ops_king_check_failed",
    "lane_d_ops_house_source_current",
    "lane_d_ops_house_source_planned",
    "lane_d_ops_house_source_installed",
    "lane_d_ops_house_source_refused",
    "lane_d_ops_king_publish_current",
    "lane_d_ops_king_publish_planned",
    "lane_d_ops_king_published",
    "lane_d_ops_king_publish_blocked",
    "lane_d_ops_king_publish_failed",
    "lane_d_ops_takedown_planned",
    "lane_d_ops_takedown_done",
    "lane_d_ops_takedown_failed",
)
