"""Miner-funded compute (DEC-CA-0036): the BYOK plumbing.

Miners fund their own challenger training leg by handing the operator a Lium
API key; the operator keeps full control of the rental, the image, and the
orchestration — the miner only pays. The pattern is ported from PRISM
(subnet 100, ``tensorlink-dev/base``: ``prism-lium-payer`` / ``lium-rent-pool``
/ ``prism-challenge`` intake), adapted to cascade's round model.

Three invariants, in order of importance:

1. **Keys are never logged, never persisted to any shared store, and never
   leave the orchestrator.** The vault holds them in memory plus (optionally)
   mode-0600 files under an operator-local directory so a control-plane
   restart can still tear down and re-rent — the same reason PRISM seals.
2. **Infra faults never burn an entry.** A dead pod, a sold-out market, or a
   429 on the miner's key re-queues the entry (retrying on the miner's kept
   key); only the miner's own key being invalid is theirs to fix, and even
   that releases the slot rather than burning it.
3. **Nothing from a miner-funded pod is trusted.** The account owner can
   always console into a pod on their own Lium account, so miner-funded legs
   receive no shared operator credentials and their checkpoints are
   provisional until the operator-funded confirmation leg (trainer policy).
"""

from .faults import (
    classify_rent_failure,
    is_auth_or_permission,
    is_no_capacity,
    is_rate_limited,
    parse_retry_secs,
    should_recover,
)
from .queue import FundedEntry, FundedQueue, rounds_needed, select_field
from .vault import PayerKeyVault

__all__ = [
    "FundedEntry",
    "FundedQueue",
    "PayerKeyVault",
    "classify_rent_failure",
    "is_auth_or_permission",
    "is_no_capacity",
    "is_rate_limited",
    "parse_retry_secs",
    "rounds_needed",
    "select_field",
    "should_recover",
]
