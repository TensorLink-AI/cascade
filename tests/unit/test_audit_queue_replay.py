"""A repeated settlement must never consume the hotkey's next funded entry."""

import pytest

from cascade.funding.queue import FundedQueue, LaneBQueue, LaneCQueue


@pytest.mark.parametrize("queue_type", [FundedQueue, LaneBQueue, LaneCQueue])
def test_replayed_completion_keeps_parked_submission(tmp_path, queue_type):
    q = queue_type(tmp_path / "queue.json")
    q.add("miner", "first", 10)
    assert q.mark_in_flight("miner", "first", target_boundary=100,
                            era_index=1, started_block=20)
    q.add("miner", "second", 30)
    q.mark_done("miner", expect_ref="first")
    assert (q.get("miner").ref, q.get("miner").status) == ("second", "queued")
    # Crash after queue completion but before the scheduler commits its state.
    restarted = queue_type(q.path)
    restarted.mark_done("miner", expect_ref="first")
    assert (restarted.get("miner").ref, restarted.get("miner").status) == ("second", "queued")
    assert restarted.mark_in_flight("miner", "second", target_boundary=200,
                                    era_index=2, started_block=120)
    restarted.mark_done("miner", expect_ref="first")
    assert restarted.get("miner").status == "in_flight"


def test_replay_does_not_finish_new_refund_of_same_ref(tmp_path):
    q = FundedQueue(tmp_path / "queue.json")
    q.add("miner", "same", 10)
    assert q.mark_in_flight("miner", "same", target_boundary=100,
                            era_index=1, started_block=20)
    q.mark_done("miner", expect_ref="same")
    q.add("miner", "same", 10)
    q.mark_done("miner", expect_ref="same")
    assert q.get("miner").status == "queued"
    assert q.mark_in_flight("miner", "same", target_boundary=200,
                            era_index=2, started_block=120)
    q.mark_done("miner", expect_ref="same", expect_started_block=20)
    assert q.get("miner").status == "in_flight"


def test_scheduler_replay_after_queue_write_keeps_next_fund(cfg, tmp_path):
    from cascade.trainer import rolling as rolling

    from .test_fix_trainer_sched_publish import _setup
    from .test_rolling import EB, FakeClient, _join

    _, clock, ops, sched, era_start, b0 = _setup(cfg, tmp_path)
    client = FakeClient()
    sched.tick(client, b0)
    _join(sched)
    queue = ops.queue()
    assert queue.get("ALFA").status == "in_flight"
    queue.add("ALFA", "next-ref", b0 + 1)
    original = ops.burn

    def die_after_queue_write(gens):
        raise RuntimeError("crash after queue completion")

    ops.burn = die_after_queue_write
    b1 = era_start + EB
    clock.t += (b1 + 3 - b0) * rolling.BLOCK_SECONDS
    with pytest.raises(RuntimeError, match="crash after queue"):
        sched.tick(client, b1 + 3)
    assert queue.get("ALFA").ref == "next-ref"
    assert queue.get("ALFA").status == "queued"
    ops.burn = original
    # Replay the persisted envelope directly; ordinary admission must not
    # interfere with this particular crash-boundary scenario.
    sched._publish_settlement(sched.state.current, sched.state.pending_publish, replay=True)
    assert queue.get("ALFA").status == "queued"
