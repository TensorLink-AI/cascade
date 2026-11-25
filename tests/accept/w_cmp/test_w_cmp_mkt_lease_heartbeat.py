"""#908 (W-CMP): the MarketCast offload lease stays fresh for the whole scoring
run, not only between rounds. A single round can forecast for longer than the
provisioner's ``mkt_request_stale_s`` (1800 s); a request touched only before
each round went stale mid-run and the pod was torn down under it."""
from __future__ import annotations

import os
import time

from cascade.validator.mkt_offload import MktLease

OLD = 3600.0


def _lease(tmp_path, cls=MktLease):
    return cls(tmp_path / "mkt_hosts.toml", host_fn=lambda: "HOST")


def _age(lease):
    old = time.time() - OLD
    os.utime(lease.request, (old, old))
    return old


def test_w_cmp_heartbeat_refreshes_the_request_during_a_long_body(tmp_path):
    lease = _lease(tmp_path)
    assert callable(getattr(lease, "heartbeat", None)), "#908: MktLease has no heartbeat"
    lease.touch("r1", force=True)
    old = _age(lease)
    with lease.heartbeat("r1", every_s=0.05):
        time.sleep(0.3)
        assert lease.request.stat().st_mtime > old + OLD - 60
    assert '"round_id": "r1"' in lease.request.read_text(encoding="utf-8")


def test_w_cmp_heartbeat_stops_after_the_body(tmp_path):
    lease = _lease(tmp_path)
    assert callable(getattr(lease, "heartbeat", None)), "#908: MktLease has no heartbeat"
    lease.touch("r1", force=True)
    with lease.heartbeat("r1", every_s=0.05):
        pass
    lease.release()
    time.sleep(0.2)
    assert not lease.request.exists()


def test_w_cmp_publisher_keeps_the_request_fresh_inside_a_long_round(tmp_path):
    """The validator's offloaded publisher holds the heartbeat across every
    round: the request is fresh WHILE a round computes, and released after."""
    from cascade.validator.marketcast import MarketCastMixin

    class _FastLease(MktLease):
        def heartbeat(self, rid="", *, every_s=0.05):  # noqa: ARG002 — fast beat for the test
            return super().heartbeat(rid, every_s=0.05)

    class _Sess:
        def __init__(self, host):
            pass

        def start(self):
            pass

        def close(self):
            pass

    class _Store:
        def get_text(self, key):
            from cascade.shared.hippius import ObjectNotFound

            raise ObjectNotFound(key)

    class V(MarketCastMixin):
        pass

    leases = []

    def _mk(path):
        leases.append(_lease(tmp_path, _FastLease))
        return leases[-1]

    v = V()
    v.mkt_eval_hosts = tmp_path / "mkt_hosts.toml"
    v.mkt_host_wait_s = 1.0
    v.mkt_lease_fn, v.mkt_session_fn = _mk, _Sess
    v._mkt_store_obj = _Store()
    fresh = []

    def _pub(item, *, holders=None):
        old = _age(leases[-1])          # the round starts with a stale request
        time.sleep(0.3)                 # ... and computes "for a long time"
        fresh.append(leases[-1].request.stat().st_mtime > old + OLD - 60)
        return True

    v._mkt_publish_round = _pub
    assert v._mkt_publish_remote([{"round_id": "r1"}, {"round_id": "r2"}], {}) == 2
    assert fresh == [True, True], "#908: the request went stale mid-round"
    assert not leases[-1].request.exists()
