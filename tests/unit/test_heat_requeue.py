"""In-round infra re-queue for heat challengers (`_run_heat_field`).

The 2026-08-19 datacenter blip killed every in-flight heat run at once;
three challengers whose runs were their dispatch retries dropped
TERMINALLY through no fault of their own. `_run_heat_field` gives
infrastructure casualties (storage layer / ssh transport) bounded
re-queues in the same heat, while miner-fault failures still drop
immediately.
"""

from __future__ import annotations

from types import SimpleNamespace

from cascade.shared.hippius import StorageError
from cascade.trainer.loop import _infra_failure, _run_heat_field


def _challenger(hotkey: str) -> SimpleNamespace:
    return SimpleNamespace(hotkey=hotkey, ref=f"ns/{hotkey}@sha256:" + "a" * 64)


def _transport_error(msg: str = "ssh died") -> RuntimeError:
    e = RuntimeError(msg)
    e.returncode = 255
    return e


def _field(run_fn, challengers, *, requeues=2, storage_dropped=None, progress=None):
    progress = progress if progress is not None else []
    return _run_heat_field(
        run_fn, challengers, max_workers=4, requeues=requeues,
        cooldown_seconds=0.0, note_progress=lambda d, t: progress.append((d, t)),
        storage_dropped=storage_dropped if storage_dropped is not None else {},
    )


def test_infra_failure_classifies_storage_and_transport_only():
    assert _infra_failure(StorageError("fetch of x failed after 1 attempt(s)"))
    assert _infra_failure(RuntimeError("cascade.shared.hippius.StorageError: chunk 0 failed"))
    assert _infra_failure(_transport_error())
    # rc=1 with the challenger's own traceback is the miner's problem.
    miner_fault = RuntimeError("remote challenger failed (rc=1): ImportError: no module")
    miner_fault.returncode = 1
    assert not _infra_failure(miner_fault)


def test_infra_casualty_is_requeued_and_recovers_in_the_same_heat():
    attempts: dict[str, int] = {}

    def run(c):
        attempts[c.hotkey] = attempts.get(c.hotkey, 0) + 1
        if c.hotkey == "flaky" and attempts[c.hotkey] == 1:
            raise StorageError("fetch of ckpt failed after 1 attempt(s): chunk 0 failed")
        return (c, f"dir-{c.hotkey}", "digest")

    progress: list = []
    out, transport = _field(run, [_challenger("flaky"), _challenger("solid")],
                            progress=progress)
    assert {c.hotkey for c, _, _ in out} == {"flaky", "solid"}
    assert attempts["flaky"] == 2
    assert transport == 0
    # Progress counts each challenger ONCE, on its terminal outcome.
    assert progress == [(1, 2), (2, 2)]


def test_requeue_budget_exhausts_to_a_terminal_storage_drop():
    def always_storage(c):
        raise StorageError("fetch failed: local I/O error")

    dropped: dict = {}
    out, transport = _field(always_storage, [_challenger("doomed")],
                            requeues=2, storage_dropped=dropped)
    assert out == []
    assert transport == 0
    # 1 dispatch + 2 re-queues, then the terminal drop keeps the burn-exemption
    # candidacy exactly as the old inline loop did.
    assert dropped == {"doomed": _challenger("doomed").ref}


def test_miner_fault_failures_never_requeue():
    calls = {"n": 0}

    def miner_bug(c):
        calls["n"] += 1
        e = RuntimeError("remote challenger failed (rc=1): ValueError: bad generator")
        e.returncode = 1
        raise e

    dropped: dict = {}
    out, transport = _field(miner_bug, [_challenger("buggy")], storage_dropped=dropped)
    assert out == []
    assert calls["n"] == 1  # no free retries for bad code
    assert dropped == {}
    assert transport == 0


def test_terminal_transport_failures_still_feed_the_wipeout_check():
    def dead_fleet(c):
        raise _transport_error()

    out, transport = _field(dead_fleet, [_challenger("a"), _challenger("b")],
                            requeues=1)
    assert out == []
    # Counted once per challenger AFTER the budget is spent — the caller's
    # all-transport 0/N wipeout guard keeps its meaning.
    assert transport == 2


def test_requeues_zero_restores_the_old_terminal_behaviour():
    calls = {"n": 0}

    def flaky(c):
        calls["n"] += 1
        raise StorageError("chunk 0 failed")

    out, _ = _field(flaky, [_challenger("x")], requeues=0)
    assert out == []
    assert calls["n"] == 1
