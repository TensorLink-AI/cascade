"""W-OFFBOX: the box's memory during a lane D settlement does not grow with
the pinned snapshots (the eval host loads them), the in-process path refuses
pins over the box budget before reading any, and the step logs its RSS with
an ERROR over the budget. Contract: README.md "Nothing heavy on the box"."""

from __future__ import annotations

import logging
import tracemalloc

import pytest

from . import helpers as h

PIN_COUNTS = (2, 8)


def _rig_with_pins(tmp_path, n: int, *, off: bool):
    """A rig whose settlement at B2 pins ``n`` copies of snapshot 2 (oldest
    first), each its own directory: the box's pool bytes grow with ``n``."""
    fx = h.w8b.fixture(tmp_path / "fx")
    rig = h.w8c.Rig(tmp_path / "rig", fx=fx)
    src = h.Path(fx["chain"]["pools"][2])
    sha = rig.pool.pins(2)["sha256"]
    box_root = tmp_path / "box_pool"
    dirs = [h.copy_pin_as(src, box_root / f"pool_d_copy{i}") for i in range(n)]
    rig.pool.override = lambda b: [{"dir": d, "sha256": sha} for d in dirs]
    if off:
        host_root = tmp_path / "host_pool"
        for d in dirs:
            h.copy_pin_as(d, host_root / d.name)
        h.offloaded(rig, host_root, fork=True)
    return rig


def _box_peak(rig) -> int:
    """Peak traced bytes in THIS process (the box) over one settlement step
    at B2; the offloaded host work runs in a forked child."""
    step = h.step(rig)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        step(h.B2)
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()


def test_box_peak_memory_stays_flat_as_pins_grow(tmp_path):
    h.require_offbox()
    for off_ in (True, False):            # one-time imports and caches, unmeasured
        _box_peak(_rig_with_pins(tmp_path / f"warm{off_}", PIN_COUNTS[0], off=off_))
    off = {n: _box_peak(_rig_with_pins(tmp_path / f"off{n}", n, off=True)) for n in PIN_COUNTS}
    ref = {n: _box_peak(_rig_with_pins(tmp_path / f"ref{n}", n, off=False))
           for n in PIN_COUNTS}
    lo, hi = PIN_COUNTS
    ref_growth = ref[hi] - ref[lo]
    off_growth = off[hi] - off[lo]
    # the measurement sees pool loads: in-process, 6 more pins cost memory
    assert ref_growth > 4 * (hi - lo) * 1024, f"in-process peaks {ref} do not grow with pins"
    assert off_growth < max(64 * 1024, ref_growth // 10), \
        f"offloaded box peaks {off} grow with the pins (in-process {ref})"
    assert off[hi] < ref[hi], f"offloaded box peak {off[hi]} >= in-process {ref[hi]}"


def test_offloaded_box_never_loads_a_pool(tmp_path, monkeypatch):
    """Every pool_d loader in this process is booby-trapped; the offloaded
    settlement still settles (the host child loads its own copy)."""
    rig = _rig_with_pins(tmp_path, 2, off=True)
    from cascade.validator import lane_d

    box = h.os.getpid()

    def trap(real):
        def load(*a, **k):
            if h.os.getpid() == box:      # the forked eval host child loads normally
                raise AssertionError("the box loaded a pool_d snapshot")
            return real(*a, **k)
        return load

    step = h.step(rig)
    for name in ("load_pool", "load_pool_for", "_load_pools"):
        monkeypatch.setattr(lane_d, name, trap(getattr(lane_d, name)))
    step(h.B2)
    assert h.report_bytes(rig, h.B2)


def test_in_process_path_refuses_pins_over_the_box_budget_before_reading(tmp_path, monkeypatch):
    h.require_offbox()
    rig = _rig_with_pins(tmp_path, 3, off=False)
    from cascade.validator import lane_d

    loads = []
    monkeypatch.setattr(lane_d, "load_pool_for", lambda *a, **k: loads.append(a))
    rig.source = rig.build(box_pool_budget_bytes=1024)
    with pytest.raises(Exception) as ei:
        rig.source(h.B2)
    h.w8c.assert_inputs_error(ei.value, "pool_unavailable")
    assert "box budget" in str(ei.value)
    assert loads == [], "the over-budget pins were read before the refusal"


def test_default_box_budget_is_far_below_the_live_pins():
    """13 live pins hold ~1.7 GB; the default in-process cap is 64 MiB."""
    h.require_offbox()
    assert h.inputs_mod().BOX_POOL_BUDGET_BYTES <= 64 * 1024 * 1024


def _worker_mod():
    m = h.importlib.import_module(h.WORKER)
    for name in ("STEP_RSS_BUDGET_BYTES", "process_rss_bytes"):
        assert hasattr(m, name), f"W-OFFBOX: offthread_worker.{name} is missing"
    return m


def test_step_rss_is_logged_and_over_budget_is_an_error(caplog):
    m = _worker_mod()
    held = []
    readings = iter([100 * 2**20] + [900 * 2**20] * 1000)

    def heavy(b):
        held.append(bytearray(2**20))

    w = m.LaneDWorker(heavy, rss_budget_bytes=512 * 2**20, rss_probe=lambda: next(readings),
                      rss_interval=0.01)
    with caplog.at_level(logging.INFO, logger=m.__name__):
        assert w.submit(5) and w.join(5)
    assert w.last_rss == {"start": 100 * 2**20, "peak": 900 * 2**20, "growth": 800 * 2**20}
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors and "box budget" in errors[0].getMessage(), "no ERROR over the budget"


def test_step_under_budget_logs_rss_without_error(caplog):
    m = _worker_mod()
    w = m.LaneDWorker(lambda b: None)
    with caplog.at_level(logging.INFO, logger=m.__name__):
        assert w.submit(6) and w.join(5)
    assert w.last_rss is not None and w.last_rss["peak"] >= w.last_rss["start"] > 0
    assert any("rss start" in r.getMessage() for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
