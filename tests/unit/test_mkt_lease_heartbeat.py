"""#908: the MarketCast offload lease stays fresh for the whole run, not only
between rounds."""
from __future__ import annotations

import os
import time

from cascade.validator.mkt_offload import MktLease


def _lease(tmp_path):
    return MktLease(tmp_path / "mkt_hosts.toml", host_fn=lambda: None)


def test_heartbeat_refreshes_the_request_during_a_long_body(tmp_path):
    lease = _lease(tmp_path)
    lease.touch("r1", force=True)
    old = time.time() - 3600
    os.utime(lease.request, (old, old))
    with lease.heartbeat("r1", every_s=0.05):
        time.sleep(0.3)
        assert lease.request.stat().st_mtime > old + 3000


def test_heartbeat_stops_after_the_body(tmp_path):
    lease = _lease(tmp_path)
    lease.touch("r1", force=True)
    with lease.heartbeat("r1", every_s=0.05):
        pass
    lease.release()
    time.sleep(0.2)
    assert not lease.request.exists()

