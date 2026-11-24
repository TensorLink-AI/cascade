"""W3 (#842): the intake refreshes the queue file on a timer, so a reveal or
a TTL expiry lands in ``lane_d_intake_queue.json`` without any request.

Testnet RC 02335f09: after a miner's ``ready`` the file kept ``ready
reveal_block=None`` until some HTTP request reached the intake; the
validator reads the file directly, so the reveal was invisible to the
settlement in a quiet period. Contract: README.md "Queue" (background
refresh). No real sleeps: the timer's ``wait`` hook drives the ticks.
"""

from __future__ import annotations

import json
import threading

from .helpers import BLOCK, TTL, Env, Miner, checkpoint, digest_of, manifest, ops_cfg


def _file_state(env: Env, hotkey: str) -> dict:
    doc = json.loads((env.state_dir / "lane_d_intake_queue.json").read_text(encoding="utf-8"))
    (rec,) = [r for r in doc["entries"] if r["hotkey"] == hotkey]
    return rec


def _ready_entry(env: Env, name: str) -> tuple[Miner, str]:
    m = Miner(env, name)
    files = checkpoint(name)
    r = m.enter(files)
    assert r.status == 200, r
    return m, digest_of(manifest(files))


def _one_tick_wait(calls: list):
    """``wait`` that records its timeout and ends the loop after one sleep."""
    def wait(timeout: float) -> bool:
        calls.append(timeout)
        return True
    return wait


def test_reveal_lands_in_the_file_after_one_tick_without_requests(tmp_path):
    env = Env(tmp_path, ops=ops_cfg(queue_poll_seconds=17))
    m, dig = _ready_entry(env, "tick-a")
    assert _file_state(env, m.hotkey)["state"] == "ready"
    env.reads.reveals[(m.hotkey, dig)] = BLOCK + 3
    env.reads.block = BLOCK + 3
    calls_before = list(env.reads.calls)
    # no request reaches the app; one timer tick
    waits: list = []
    timer = env.app.refresh_timer(wait=_one_tick_wait(waits))
    timer.run()
    assert timer.ticks == 1
    assert waits == [17.0]                      # [lane_d_ops] queue_poll_seconds
    rec = _file_state(env, m.hotkey)
    assert rec["state"] == "revealed" and rec["reveal_block"] == BLOCK + 3
    assert env.reads.calls[len(calls_before):][:1] == ["current_block"]


def test_ttl_expiry_lands_in_the_file_after_one_tick(tmp_path):
    env = Env(tmp_path)
    m, _dig = _ready_entry(env, "tick-b")
    env.reads.block = BLOCK + TTL + 1
    assert env.app.tick() == BLOCK + TTL + 1
    assert _file_state(env, m.hotkey)["state"] == "ready_expired"


def test_startup_refresh_on_restart(tmp_path):
    env = Env(tmp_path)
    m, dig = _ready_entry(env, "tick-c")
    env.reads.reveals[(m.hotkey, dig)] = BLOCK + 2
    env.reads.block = BLOCK + 2
    app = env.restart()
    app.tick()                                  # main() ticks before it binds
    assert _file_state(env, m.hotkey)["state"] == "revealed"


def test_failed_chain_read_skips_the_tick_and_keeps_the_entry(tmp_path):
    env = Env(tmp_path)
    m, dig = _ready_entry(env, "tick-d")
    env.reads.reveals[(m.hotkey, dig)] = BLOCK + 1
    env.reads.block = BLOCK + TTL + 5
    env.reads.fail.add("current_block")
    assert env.app.tick() is None
    assert _file_state(env, m.hotkey)["state"] == "ready"
    env.reads.fail.discard("current_block")
    env.reads.fail.add("reveal_block")
    env.app.tick()                              # a failed reveal read never expires
    assert _file_state(env, m.hotkey)["state"] == "ready"
    env.reads.fail.discard("reveal_block")
    env.app.tick()
    assert _file_state(env, m.hotkey)["state"] == "revealed"


def test_timer_survives_an_unexpected_error_and_keeps_ticking(tmp_path):
    env = Env(tmp_path)
    m, dig = _ready_entry(env, "tick-e")
    boom = {"n": 0}
    real_tick = env.app.tick

    def flaky_tick():
        boom["n"] += 1
        if boom["n"] == 1:
            raise RuntimeError("unexpected")
        return real_tick()

    env.app.tick = flaky_tick
    env.reads.reveals[(m.hotkey, dig)] = BLOCK + 4
    env.reads.block = BLOCK + 4
    waits: list = []

    def wait(timeout: float) -> bool:
        waits.append(timeout)
        return len(waits) >= 2
    timer = env.app.refresh_timer(60, wait=wait)
    timer.run()
    assert timer.ticks == 2
    assert _file_state(env, m.hotkey)["state"] == "revealed"


def test_timer_shares_the_request_lock(tmp_path):
    env = Env(tmp_path)
    _ready_entry(env, "tick-f")
    entered = threading.Event()
    done = threading.Event()
    with env.app._lock:
        t = threading.Thread(target=lambda: (entered.set(), env.app.tick(), done.set()))
        t.start()
        assert entered.wait(5)
        assert not done.wait(0.05)               # blocked behind the held lock
    assert done.wait(5)
    t.join(5)


def test_timer_start_stop_thread(tmp_path):
    env = Env(tmp_path)
    timer = env.app.refresh_timer(3600).start()
    timer.stop(timeout=5)
    assert not timer._thread.is_alive()
    assert timer.ticks >= 1


def test_interval_must_be_positive(tmp_path):
    import pytest

    from cascade.lane_d.service import RefreshTimer

    env = Env(tmp_path)
    for bad in (0, -1, True, float("nan"), "60", None):
        with pytest.raises(ValueError):
            RefreshTimer(env.app, bad)
