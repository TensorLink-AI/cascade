"""The FIFO telemetry thread gives the inline accumulator's exact state."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from cascade.trainer.channel_stats import BackgroundObserver, ChannelStatsAccumulator


def _series(n=60):
    rng = np.random.default_rng(5)
    out = []
    for i in range(n):
        c = 1 + i % 5
        a = rng.standard_normal((c, 64)).cumsum(-1)
        if i % 3 == 0:
            out.append({"values": a, "groups": np.arange(c) % 2})
        else:
            out.append(a)
        if i % 7 == 0:
            out.append(out[-1])                   # duplicates exercise the dedup
    return out


def _state(acc):
    return (acc._corrs, acc._ranks, acc._partner_z, acc._n_channels, acc._seen, acc.summary())


def test_background_observer_matches_inline_bit_for_bit():
    inline, threaded = ChannelStatsAccumulator(), ChannelStatsAccumulator()
    bg = BackgroundObserver(threaded.observe, maxsize=4)  # small queue: back-pressure path
    for item in _series():
        inline.observe(item)
        bg(item)
    bg.close()
    a, b = _state(inline), _state(threaded)
    assert a[:4] == b[:4]                          # same floats, same ORDER
    assert a[4] == b[4] and a[5] == b[5] and a[5] is not None


def test_observer_order_is_fifo_and_flush_waits():
    seen, gate = [], threading.Event()

    def slow(x):
        gate.wait(5)
        seen.append(x)

    bg = BackgroundObserver(slow, maxsize=100)
    for i in range(50):
        bg(i)
    gate.set()
    bg.flush()
    assert seen == list(range(50))
    bg.close()
    bg.close()                                     # idempotent


def test_observer_failure_surfaces_on_the_training_thread():
    def boom(x):
        if x == 2:
            raise ValueError("bad series")

    bg = BackgroundObserver(boom)
    for i in range(3):
        bg(i)
    with pytest.raises(ValueError, match="bad series"):
        bg.flush()
    with pytest.raises(ValueError, match="bad series"):
        bg(9)
    with pytest.raises(ValueError, match="bad series"):
        bg.close()
    bg.close(raise_errors=False)
    with pytest.raises(RuntimeError, match="closed"):
        bg(1)
