"""Shared pieces of the lane B / C / D full-cycle tests (W-E2E).

* :class:`Cycle` records each stage a lane's cycle passes through, so a test
  asserts the WHOLE path ran (submit ... miner-visible status), not just the
  last stage.
* :func:`box_guard` is the box-side import tripwire: inside it, importing
  ``torch``, ``timestral``, ``safetensors``, ``transformers`` or ``triton`` at
  CALL time raises :class:`BoxImportRefused`. The existing torch-free guard
  (tests/unit/test_box_side_torch_free.py) only checks import time; #1003's
  timestral import sat inside a function.
* :func:`box_budget` measures the Python heap peak (tracemalloc, which numpy
  reports to) over a box stage and fails when it exceeds a budget: the 7 GB
  box OOMs of 2026-11-28 (lane D pool pins, MarketCast covariate rows) were
  in-process loads of data meant for a pod. Fixtures are sized so the pod
  path stays far under the budget and an in-process load goes far over it.
* :class:`BlockClock` is a compressed-epoch chain clock with deterministic
  block hashes (no chain).
* :func:`revert_patch` applies a named pre-fix behaviour ("revert patch") for
  a bug already fixed on main, so a test can show the harness WOULD catch it.

Nothing here imports cascade at module level."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.abc
import sys
import tracemalloc
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

# Modules that must never load in a box process (validator, trainer, intake).
BOX_FORBIDDEN = ("torch", "timestral", "safetensors", "transformers", "triton")


class BoxImportRefused(ModuleNotFoundError):
    """A box-side stage imported a GPU / model library."""


class _BoxFinder(importlib.abc.MetaPathFinder):
    def __init__(self, forbidden: tuple[str, ...], hits: list[str]):
        self.forbidden = forbidden
        self.hits = hits

    def find_spec(self, name, path, target=None):
        root = name.split(".", 1)[0]
        if root in self.forbidden:
            frame = sys._getframe(1)
            while frame is not None and frame.f_globals.get("__name__", "").startswith("importlib"):
                frame = frame.f_back
            where = (f"{frame.f_code.co_filename}:{frame.f_lineno}" if frame is not None
                     else "?")
            self.hits.append(f"{name} from {where}")
            raise BoxImportRefused(f"box-side import of {name!r} refused (from {where})",
                                   name=name)
        return None


@contextlib.contextmanager
def box_guard(forbidden: tuple[str, ...] = BOX_FORBIDDEN) -> Iterator[list[str]]:
    """Refuse every import of ``forbidden`` (and submodules) inside the block.

    Already-imported copies are hidden from ``sys.modules`` for the duration
    (a test run on a box whose venv has torch must still trip), then restored.
    Yields the list of refused imports (``"<module> from <file:line>"``)."""
    hits: list[str] = []
    hidden = {k: sys.modules.pop(k) for k in list(sys.modules)
              if k.split(".", 1)[0] in forbidden}
    finder = _BoxFinder(forbidden, hits)
    sys.meta_path.insert(0, finder)
    try:
        yield hits
    finally:
        sys.meta_path.remove(finder)
        for k in [k for k in sys.modules if k.split(".", 1)[0] in forbidden]:
            sys.modules.pop(k, None)
        sys.modules.update(hidden)


class BoxBudgetExceeded(AssertionError):
    """A box stage's heap peak went over its budget."""


@dataclass
class BudgetReading:
    label: str
    budget: int
    peak: int = 0


@contextlib.contextmanager
def box_budget(label: str, budget_bytes: int) -> Iterator[BudgetReading]:
    """Fail with :class:`BoxBudgetExceeded` when the traced heap peak over the
    block exceeds ``budget_bytes``."""
    reading = BudgetReading(label, int(budget_bytes))
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base, _ = tracemalloc.get_traced_memory()
    try:
        yield reading
    finally:
        _, peak = tracemalloc.get_traced_memory()
        reading.peak = max(0, peak - base)
        if started:
            tracemalloc.stop()
    if reading.peak > reading.budget:
        raise BoxBudgetExceeded(f"box stage {label!r}: heap peak {reading.peak} B over "
                                f"its {reading.budget} B budget (a pod-side load ran in-process)")


@dataclass
class BlockClock:
    """A compressed-epoch chain clock: ``epoch_blocks`` per settlement slot."""

    block: int
    epoch_blocks: int = 10

    def advance(self, n: int = 1) -> int:
        self.block += int(n)
        return self.block

    def next_boundary(self) -> int:
        e = self.epoch_blocks
        return (self.block // e + 1) * e

    def to_next_boundary(self) -> int:
        self.block = self.next_boundary()
        return self.block

    @staticmethod
    def block_hash(block: int) -> str:
        return "0x" + hashlib.sha256(f"e2e-block-{int(block)}".encode()).hexdigest()


@dataclass
class Stage:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Cycle:
    """The stages one lane cycle passed through, in order."""

    lane: str
    stages: list[Stage] = field(default_factory=list)

    def ran(self, name: str, detail: str = "") -> None:
        self.stages.append(Stage(name, True, detail))

    def names(self) -> list[str]:
        return [s.name for s in self.stages if s.ok]

    def assert_full(self, expected: list[str] | tuple[str, ...]) -> None:
        got = self.names()
        missing = [s for s in expected if s not in got]
        assert not missing, f"lane {self.lane} cycle stopped before {missing}; ran {got}"
        order = [got.index(s) for s in expected]
        assert order == sorted(order), f"lane {self.lane} stages out of order: {got}"


@contextlib.contextmanager
def revert_patch(monkeypatch_ctx, target: object, attr: str, pre_fix: Callable) -> Iterator[None]:
    """Swap ``target.attr`` for its pre-fix behaviour inside the block (the
    'revert patch' for a bug already fixed on main)."""
    with monkeypatch_ctx.context() as mp:
        mp.setattr(target, attr, pre_fix)
        yield
