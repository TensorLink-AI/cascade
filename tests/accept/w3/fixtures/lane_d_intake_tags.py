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
