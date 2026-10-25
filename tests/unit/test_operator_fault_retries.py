"""Work lost on our side retries on the payer's key a bounded number of times."""
from __future__ import annotations

import pytest

from cascade.funding.queue import OPERATOR_FAULT_RETRIES, FundedQueue, LaneBQueue, LaneCQueue

REF = "cascade/gen@sha256:" + "a" * 64


def test_each_operator_side_loss_counts_once_and_survives_restart(tmp_path):
    q = FundedQueue(tmp_path / "fq.json", entry_ttl_seconds=3600.0)
    q.add("hk", REF, reveal_block=1)
    assert q.get("hk").operator_faults == 0
    q.set_operator_billed("hk", True)
    q.set_operator_billed("hk", True)          # the same loss flagged twice
    q.requeue("hk", error="lost", error_class="infra", burn_attempt=False,
              operator_billed=True)            # ... and requeued with the flag
    assert q.get("hk").operator_faults == 1
    q.set_operator_billed("hk", False)         # retried on the payer's key
    q.requeue("hk", error="lost again", error_class="no_capacity", burn_attempt=False,
              operator_billed=True)
    reloaded = FundedQueue(tmp_path / "fq.json", entry_ttl_seconds=3600.0).get("hk")
    assert reloaded.operator_faults == 2 and reloaded.operator_billed
    assert reloaded.attempts == 0              # never burned
    assert OPERATOR_FAULT_RETRIES == 3


@pytest.mark.parametrize("lane", ["a", "b", "c"])
@pytest.mark.parametrize("faults", [0, 1, 2, 3, 4])
def test_payer_rental_stops_at_three_operator_losses(tmp_path, monkeypatch, lane, faults):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from cascade.provision import funded
    from cascade.trainer.faults import _FundedLegSkip
    from tests.unit.test_funded_expired_key_operator_rent import _expired_runner
    from tests.unit.test_funded_pod_wiring import _challenger, _vault

    runner, _ = _expired_runner(tmp_path)
    _vault(tmp_path, "hkA")
    runner._leg_local_obj = SimpleNamespace(operator_rent_ok=False)
    queue = {"a": FundedQueue, "b": LaneBQueue, "c": LaneCQueue}[lane](tmp_path / "lane.json")
    if lane == "b":
        queue.add_lane_b("hkA", ("energy",), REF, 1)
    elif lane == "c":
        queue.add_lane_c("hkA", "mkt:returns", REF, 1)
    else:
        queue.add("hkA", REF, 1)
    for _ in range(faults):
        queue.set_operator_billed("hkA", False)
        queue.set_operator_billed("hkA", True)
    runner._funded_queue = lambda: queue

    class PayerReached(Exception):
        pass

    rent = Mock(side_effect=PayerReached)
    monkeypatch.setattr(funded, "rent_funded_pod", rent)
    if faults >= OPERATOR_FAULT_RETRIES:
        with pytest.raises(_FundedLegSkip):
            runner._rent_funded_host("777", _challenger("hkA", ref=REF))
        rent.assert_not_called()
        assert "paused after" in runner._funded_leg_failures["hkA"][0]
        assert queue.get("hkA").operator_billed
    else:
        with pytest.raises(PayerReached):
            runner._rent_funded_host("777", _challenger("hkA", ref=REF))
        rent.assert_called_once()
        assert not queue.get("hkA").operator_billed
    assert queue.get("hkA").attempts == 0


def test_loss_limit_still_allows_configured_operator_recovery(tmp_path):
    from cascade.trainer.faults import _FundedOperatorRent
    from tests.unit.test_funded_expired_key_operator_rent import _expired_runner
    from tests.unit.test_funded_pod_wiring import _challenger, _vault

    runner, queue = _expired_runner(tmp_path)
    _vault(tmp_path, "hkA")
    for _ in range(OPERATOR_FAULT_RETRIES):
        queue.set_operator_billed("hkA", False)
        queue.set_operator_billed("hkA", True)
    with pytest.raises(_FundedOperatorRent):
        runner._rent_funded_host("777", _challenger("hkA"))
    assert queue.get("hkA").operator_billed and queue.get("hkA").attempts == 0
