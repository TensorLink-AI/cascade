"""Seed-mix read-ahead (``_Prefetch``): the producer thread must exit promptly
once the consumer is gone — a leg that hits its wall stops consuming with the
queue full, and ``close()`` must not leave a thread blocked forever on the end
marker or a raised exception, nor close a child under a live producer."""

from __future__ import annotations

import threading
import time

import numpy as np

from cascade.trainer import stream as stream_mod
from cascade.trainer.stream import _Prefetch, _SeedMixStream


def _wait_dead(pf: _Prefetch, limit: float = 1.0) -> bool:
    return pf.join(limit)


def test_full_queue_then_consumer_stops_thread_exits():
    pf = _Prefetch(iter(range(10)), maxsize=2)       # producer ends with a full queue
    time.sleep(0.2)
    assert pf.alive                                  # blocked on the queue
    pf.stop()
    assert _wait_dead(pf)


def test_end_marker_put_is_stop_aware():
    # 2 items fill the queue exactly; the producer then blocks on the END put
    pf = _Prefetch(iter([1, 2]), maxsize=2)
    time.sleep(0.2)
    pf.stop()
    assert _wait_dead(pf)


def test_raising_iterator_with_full_queue_exits():
    def gen():
        yield 1
        yield 2
        raise RuntimeError("generator died")

    pf = _Prefetch(gen(), maxsize=2)
    time.sleep(0.2)
    pf.stop()
    assert _wait_dead(pf)


def test_exception_still_reaches_a_live_consumer():
    def gen():
        yield 1
        raise RuntimeError("boom")

    pf = _Prefetch(gen(), maxsize=1)
    assert next(pf) == 1
    try:
        next(pf)
    except RuntimeError as e:
        assert "boom" in str(e)
    else:  # pragma: no cover
        raise AssertionError("exception not surfaced")
    assert _wait_dead(pf)


class _Child:
    """A child stream whose generator records concurrent use with close()."""

    def __init__(self, seed: int, raise_after: int | None = None) -> None:
        self.seed, self.raise_after = seed, raise_after
        self.closed = False
        self.used_after_close = False
        self.lock = threading.Lock()

    def series(self):
        rng = np.random.default_rng(self.seed)
        i = 0
        while True:
            if self.closed:
                self.used_after_close = True
                return
            if self.raise_after is not None and i >= self.raise_after:
                raise RuntimeError("child failed")
            i += 1
            yield rng.normal(size=(1, 8))

    def close(self) -> None:
        self.closed = True


def test_seed_mix_close_after_early_stop_joins_producers(monkeypatch):
    monkeypatch.setattr(stream_mod, "PREFETCH_SERIES", 4)
    children = [_Child(i) for i in range(3)]
    s = _SeedMixStream(children)
    it = s.series()
    for _ in range(5):                       # consumer stops early (the wall)
        next(it)
    t0 = time.monotonic()
    s.close()
    assert time.monotonic() - t0 < 1.5
    assert all(not pf.alive for pf in s._prefetch)
    # producers left the child iterators BEFORE the children were closed
    assert not any(c.used_after_close for c in children)


def test_seed_mix_close_with_a_raising_child():
    children = [_Child(0), _Child(1, raise_after=1)]
    s = _SeedMixStream(children)
    it = s.series()
    next(it)
    time.sleep(0.2)
    s.close()
    assert all(not pf.alive for pf in s._prefetch)


def test_seed_mix_order_unchanged_by_prefetch():
    """The interleave is one series per child per turn — the same order (and
    so the same rolling digest) as reading the children directly."""
    def direct(n):
        gens = [_Child(i).series() for i in range(3)]
        out = []
        while len(out) < n:
            for g in gens:
                out.append(next(g))
        return out[:n]

    s = _SeedMixStream([_Child(i) for i in range(3)])
    it = s.series()
    got = [next(it) for _ in range(12)]
    s.close()
    assert all(np.array_equal(a, b) for a, b in zip(got, direct(12), strict=True))
