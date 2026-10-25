"""Cross-service behaviour preserved by the shared funding/status/deadline helpers."""

import threading
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from cascade.funding.queue import FundedQueue, LaneBQueue, LaneCQueue
from cascade.miner.funding import FundingRequest
from cascade.provision.loop import ProvisionerLoop
from cascade.shared.concurrency import call_with_deadline
from cascade.trainer import lane_b_driver, lane_c_driver, lane_status
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.runner_common import funded_queue_of


@pytest.mark.parametrize("call", [call_with_deadline, TrainerRunner._with_deadline,
                                 ProvisionerLoop._with_deadline])
def test_deadline_preserves_result_error_and_nonblocking_timeout(call):
    value = object()
    assert call(lambda: value, 1) is value
    error = ValueError("fixture failure")

    def failed():
        raise error

    with pytest.raises(ValueError) as caught:
        call(failed, 1)
    assert caught.value is error

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def blocked():
        entered.set()
        release.wait(3)
        finished.set()

    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            call(blocked, .025)
        assert entered.is_set() and not finished.is_set()
        assert time.monotonic() - start < 1
    finally:
        release.set()
        assert finished.wait(1)


@pytest.mark.parametrize("absolute", [True, False])
def test_lane_queue_paths_ttl_and_fresh_reads_remain_isolated(tmp_path, absolute):
    path = tmp_path / "custom" / "a.json" if absolute else "custom/a.json"
    runner = SimpleNamespace(work_root=tmp_path, cfg=SimpleNamespace(round=SimpleNamespace(
        funded_mode="miner", funded_queue_path=path, funded_entry_ttl_hours=7)))
    qa = funded_queue_of(runner)
    qb = lane_b_driver.lane_b_queue_of(runner)
    qc = lane_c_driver.lane_c_queue_of(runner)
    assert type(qa) is FundedQueue and type(qb) is LaneBQueue and type(qc) is LaneCQueue
    assert qa.path == tmp_path / "custom" / "a.json"
    assert qb.path == tmp_path / "custom" / "lane_b_queue.json"
    assert qc.path == tmp_path / "custom" / "lane_c_queue.json"
    assert {q.entry_ttl_seconds for q in (qa, qb, qc)} == {7 * 3600}
    qa.add("alice", "a/gen@sha256:" + "a" * 64, 1)
    qb.add_lane_b("bob", ("energy",), "b/gen@sha256:" + "b" * 64, 2)
    qc.add_lane_c("carol", "mkt:returns", "c/gen@sha256:" + "c" * 64, 3)
    assert [e.hotkey for e in funded_queue_of(runner).entries()] == ["alice"]
    assert [e.hotkey for e in lane_b_driver.lane_b_queue_of(runner).entries()] == ["bob"]
    assert [e.hotkey for e in lane_c_driver.lane_c_queue_of(runner).entries()] == ["carol"]
    assert funded_queue_of(runner) is not qa


def test_funding_off_does_not_resolve_paths_or_require_other_settings():
    runner = SimpleNamespace(cfg=SimpleNamespace(round=SimpleNamespace(funded_mode="off")))
    assert funded_queue_of(runner) is None
    assert lane_b_driver.lane_b_queue_of(runner) is None
    assert lane_c_driver.lane_c_queue_of(runner) is None


def test_lane_status_heartbeats_failures_and_clocks_are_independent(monkeypatch):
    import cascade.shared.hippius as storage

    sent = []
    clock = [100.0]
    monkeypatch.setattr(lane_b_driver, "_STATUS_LAST", {})
    monkeypatch.setattr(lane_c_driver, "_STATUS_LAST", {})
    monkeypatch.setattr(lane_status, "datetime", SimpleNamespace(
        now=lambda tz: datetime.fromtimestamp(1_700_000_000 + clock[0], tz)))
    monkeypatch.setattr(storage, "publish_public_json", lambda store, key, doc: sent.append((key, doc)))
    driver = SimpleNamespace(status=lambda: {"open": [], "queue": {}})
    sched = SimpleNamespace(lane_b=driver, lane_c=driver,
                            r=SimpleNamespace(manifest_store=lambda: object()))
    publish_b, publish_c = lane_b_driver.publish_lane_b_status, lane_c_driver.publish_lane_c_status
    kwargs = {"clock": lambda: clock[0]}
    assert publish_b(sched, 900, **kwargs)
    assert publish_c(sched, 900, **kwargs)
    assert [key for key, _ in sent] == ["status/lane_b.json", "status/lane_c.json"]
    assert [doc["lane"] for _, doc in sent] == ["b", "c"]
    assert not publish_b(sched, 900, **kwargs)
    assert not publish_c(sched, 900, **kwargs)
    clock[0] += 60
    assert publish_b(sched, 900, **kwargs)  # unchanged block/body still has a fresh heartbeat
    assert sent[-1][1]["as_of"] != sent[0][1]["as_of"]

    def unavailable(*args):
        raise OSError("fixture storage outage")

    monkeypatch.setattr(storage, "publish_public_json", unavailable)
    assert not publish_c(sched, 900, **kwargs)
    monkeypatch.setattr(storage, "publish_public_json", lambda store, key, doc: sent.append((key, doc)))
    assert not publish_c(sched, 900, **kwargs)  # existing retry rate limit survives the outage
    clock[0] += 60
    assert publish_c(sched, 900, **kwargs)
    assert sent[-1][1]["as_of"] == datetime.fromtimestamp(1_700_000_000 + clock[0], UTC).isoformat()
    assert not publish_b(SimpleNamespace(), 900, **kwargs)


def test_funding_request_representation_does_not_expose_credentials():
    request = FundingRequest("https://fixture.invalid", "public-hotkey", lambda m: m,
                             "fixture-only-key", "label", ("H100", False, 1.0))
    assert "fixture-only-key" not in repr(request)
    assert "sign_fn" not in repr(request)
