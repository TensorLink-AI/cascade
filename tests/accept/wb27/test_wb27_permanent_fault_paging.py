"""Permanent-fault paging (retro 2026-11-28 item #6, class C4).

A fault that cannot heal by retrying (a library missing in the process that
runs the step, a GPU arch the pinned torch has no kernels for, a contract or
image pin mismatch) must stop its in-window retries and PAGE the owner — not
defer quietly. Lane C's timestral ImportError was deferred for ~58 h as
"MarketCast report absent"; the 01:27 lane B judge died on sm_120 ≠ torch
2.4.1 and looked like any other failed settlement.

The rules pinned here:

* :mod:`cascade.shared.permanent_fault` names permanent faults by exception
  type (cause chain included) and by a short marker list; transient faults
  (timeouts, refused connections, 5xx, a short pod disk) stay ``None``;
* a permanent lane B / lane C settlement failure is NOT retried in-window:
  lane A publishes at once without it and a ``PERMANENT FAULT`` line is
  logged at CRITICAL (cascade-ops pages on it); transient failures keep the
  #993 in-window retries unchanged;
* the same error spending ``SETTLE_REPEAT_THRESHOLD`` settlements in a row
  pages too (the next-settlement retry continues — it costs nothing).

No torch: the trainer's fakes from ``tests/unit/test_rolling.py``.
"""

from __future__ import annotations

import logging

from tests.accept.wb27.test_wb27_in_window_settlement_retry import _LaneB, _LaneC, _Run


class TimestralUnavailable(ImportError):
    """Same name as cascade.eval.timestral_adapter's (no torch import here)."""


def test_classify_names_this_weeks_permanent_faults():
    from cascade.shared.permanent_fault import classify, classify_text

    pf = classify(TimestralUnavailable("timestral is not installed in this process"))
    assert pf is not None and pf.kind == "missing_library"
    pf = classify(ModuleNotFoundError("No module named 'timestral'"))
    assert pf is not None and pf.kind == "missing_library"
    # the 2026-11-28 01:44 lane B judge on a Blackwell pod
    pf = classify(RuntimeError("CUDA error: no kernel image is available for execution "
                               "on the device"))
    assert pf is not None and pf.kind == "gpu_arch_unsupported"
    # wrapped: the cause chain decides
    try:
        try:
            raise ModuleNotFoundError("No module named 'timestral'")
        except ImportError as inner:
            raise RuntimeError("judge failed") from inner
    except RuntimeError as outer:
        pf = classify(outer)
    assert pf is not None and pf.kind == "missing_library"
    # a pod judge's remote traceback text
    pf = classify_text("ModuleNotFoundError: No module named 'timestral'")
    assert pf is not None and pf.kind == "missing_library" and pf.rule
    # transient faults stay transient
    for e in (OSError("pod disk short (attempt 1)"), OSError("hippius 503"),
              TimeoutError("timed out waiting for the pod"),
              ValueError("pod reply is not JSON")):
        assert classify(e) is None, e
    assert classify_text("ConnectionError: 502 bad gateway; no module named x") is None


def test_repeat_tracker_folds_numbers_and_resets():
    from cascade.shared.permanent_fault import RepeatTracker

    t = RepeatTracker(3)
    assert t.failed("s", "pod disk short: 41 GB free, need 60 GB") is None
    assert t.failed("s", "pod disk short: 39 GB free, need 60 GB") is None
    pf = t.failed("s", "pod disk short: 12 GB free, need 60 GB")
    assert pf is not None and pf.kind == "repeated"
    t.ok("s")
    assert t.failed("s", "pod disk short: 1 GB free, need 60 GB") is None
    assert t.failed("s", "something else") is None and t.count("s") == 1


class _LaneBMissingLib(_LaneB):
    def settle(self, *, round_id, block, era_index, king_pointer, work_dir, **kw):
        self.calls.append({"round_id": round_id})
        raise ModuleNotFoundError("No module named 'timestral'")


class _LaneCMissingLib(_LaneC):
    def settle(self, **kw):
        self.puts += 1
        raise TimestralUnavailable("timestral is not installed in this process")


def _pages(caplog):
    return [r for r in caplog.records
            if "PERMANENT FAULT" in r.getMessage() and r.levelno >= logging.CRITICAL]


def test_permanent_lane_b_failure_pages_and_is_not_retried_in_window(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    run = _Run(tmp_path, lane_b=_LaneBMissingLib())
    # lane A did not wait for an in-window retry that cannot succeed
    (m,) = run.until_published(limit=600.0)
    assert m.expert_manifest_sha256 == ""
    assert run.clock.t - run.t0 <= 60.0
    for _ in range(10):
        run.step()
    assert len(run.lane_b.calls) == 1, "a permanent fault was retried in-window"
    pages = _pages(caplog)
    assert pages, "no PERMANENT FAULT page was logged"
    msg = pages[0].getMessage()
    assert "[missing_library]" in msg and "lane B settlement" in msg and "owner paged" in msg


def test_transient_lane_b_failure_still_retries_in_window(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    run = _Run(tmp_path, lane_b=_LaneB(fails=1))
    (m,) = run.until_published()
    assert m.expert_manifest_sha256 != ""
    assert len(run.lane_b.calls) == 2
    assert not _pages(caplog)


def test_permanent_lane_c_failure_pages_and_is_not_retried_in_window(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    lane_c = _LaneCMissingLib()
    run = _Run(tmp_path, lane_c=lane_c)
    (m,) = run.until_published(limit=600.0)
    assert run.clock.t - run.t0 <= 60.0
    for _ in range(10):
        run.step()
    assert lane_c.puts == 1, "a permanent lane C fault was retried in-window"
    assert any("lane C settlement" in r.getMessage() for r in _pages(caplog))


def test_same_error_over_settlements_pages(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    run = _Run(tmp_path / "r")
    sched = run.sched
    n = int(sched.SETTLE_REPEAT_THRESHOLD)
    for i in range(n - 1):
        sched._settle_fault_final("lane B settlement", ValueError(f"pod reply is not JSON "
                                                                  f"(byte {i})"))
    assert not _pages(caplog)
    sched._settle_fault_final("lane B settlement", ValueError("pod reply is not JSON "
                                                              "(byte 99)"))
    pages = _pages(caplog)
    assert pages and "[repeated]" in pages[-1].getMessage()
