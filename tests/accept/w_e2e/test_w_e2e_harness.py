"""W-E2E harness guards: the box import tripwire fires at CALL time (the
#1003 class: a function-local ``import timestral`` passed the import-time
torch-free test), even when the venv already imported torch, and the box
budget catches an in-process load of pod-sized data (the 2026-11-28 OOMs)."""

from __future__ import annotations

import sys
import types

import pytest

from tests.e2e_lanes.harness import (
    BOX_FORBIDDEN,
    BlockClock,
    BoxBudgetExceeded,
    BoxImportRefused,
    Cycle,
    box_budget,
    box_guard,
)


def _settle_like():
    """A settlement step with a lazy, call-time import (resolve_pinned_base pre-#1003)."""
    import timestral  # noqa: F401

    return "verdict"


def test_call_time_import_is_refused_and_named():
    with box_guard() as hits, pytest.raises(BoxImportRefused, match="timestral"):
        _settle_like()
    assert hits and hits[0].startswith("timestral from ")
    assert __file__.split("/")[-1] in hits[0]


def test_an_already_imported_module_still_trips(monkeypatch):
    fake = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake)
    with box_guard(), pytest.raises(BoxImportRefused):
        import torch  # noqa: F401
    assert sys.modules["torch"] is fake            # restored afterwards


def test_every_forbidden_root_and_submodule():
    for name in BOX_FORBIDDEN:
        with box_guard(), pytest.raises(BoxImportRefused):
            __import__(f"{name}.sub")


def test_box_budget_catches_an_in_process_load():
    import numpy as np

    with pytest.raises(BoxBudgetExceeded, match="pool"), box_budget("pool", 1 << 20):
        np.ones(8 << 20, dtype=np.uint8)
    with box_budget("pointer", 1 << 20) as r:
        np.ones(1024, dtype=np.uint8)
    assert r.peak < 1 << 20


def test_clock_and_cycle():
    c = BlockClock(block=13, epoch_blocks=10)
    assert c.next_boundary() == 20 and c.to_next_boundary() == 20 and c.next_boundary() == 30
    assert BlockClock.block_hash(20) == BlockClock.block_hash(20) != BlockClock.block_hash(21)
    cyc = Cycle("x")
    cyc.ran("submit")
    cyc.ran("settle")
    cyc.assert_full(["submit", "settle"])
    with pytest.raises(AssertionError, match="stopped before"):
        cyc.assert_full(["submit", "settle", "status"])
    with pytest.raises(AssertionError, match="out of order"):
        cyc.assert_full(["settle", "submit"])
