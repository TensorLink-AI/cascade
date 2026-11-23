"""Shared spec and fixtures for the W19 acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md:

- W19 row: "Lane D work runs **off the weight/poll thread**; the weight-loop
  keep-alive (#471) is in the release (invariant 6, T15)". Accept:
  "`scripts/laned/accept W19` (a fixture slow lane D step leaves weight-set
  timing unchanged)".
- §3.2 invariant 6: "**Weights never wait for lane D** (off the poll thread;
  keep-alive). **(rev3)** #471 is merged (0c52d07) but not deployed (rc37
  lacks it); it ships with S2b. (W19, X3 T15)"
- §4 T15: "2 h lane D sweep across weight intervals | No weight gap >
  `weight_set_interval_blocks`".

The chain is the only external service: :class:`FakeChain` is the
``ChainClient`` duck type the weight loop calls (block clock, metagraph,
``set_equal_share_weights`` / ``set_weight_vector``), no real chain. The
lane D step is a fixture (a hung step, or a real ``SweepSession`` over a
bench host that hangs). Nothing here imports a module in PATHS at import
time.
"""

from __future__ import annotations

import threading
import time
import tomllib
from dataclasses import fields, replace
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Generic off-thread infrastructure, deliberately NOT named lane_d_*: W9's
# invariant 12 (tests/accept/w9/helpers.py FORBIDDEN_STEMS) forbids the
# audit's import closure, which reaches cascade/validator/loop.py, from
# loading any `cascade.validator.lane_d_*` module, and loop.py imports this.
WORKER_MODULE = "cascade.validator.offthread_worker"
# What the worker module must never import (W9 invariant 12, via loop.py).
WORKER_FORBIDDEN_PREFIX = "cascade.validator.lane_d"

KING = "w19KingHotkey"
ARM_BLOCK = 9_720_000           # an era boundary on the 900-block grid
SETTLEMENT = 900
B1 = ARM_BLOCK + SETTLEMENT     # the first armed lane D settlement
B2 = ARM_BLOCK + 2 * SETTLEMENT
INTERVAL = 180                  # [validator] weight_set_interval_blocks (mainnet)
POLL_BLOCKS = 10                # poll_seconds = 120 at 12 s blocks
TICK_BUDGET_S = 1.0             # one poll tick's wall-clock budget
HANG_S = 60.0                   # a hung step's own safety release

ARMED_LANE_D = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
                "remainder": "burn"}


@cache
def _shipped() -> str:
    return (REPO / "chain.toml").read_text(encoding="utf-8")


def chain_cfg(state_dir: Path, *, armed: bool):
    """The shipped mainnet config (real loader parse); ``armed`` arms
    ``[lane_d]`` at :data:`ARM_BLOCK`. Validator state paths point into
    ``state_dir`` so nothing is read or written outside the test's tmp dir."""
    from cascade.shared.config import parse_chain_config

    raw = tomllib.loads(_shipped())
    ld = dict(raw.get("lane_d") or {})
    if armed:
        ld.update(ARMED_LANE_D)
    raw["lane_d"] = ld
    cfg = parse_chain_config(raw)
    v = replace(cfg.validator, weight_set_interval_blocks=INTERVAL, force_burn=False,
                state_db_path=str(Path(state_dir) / "validator_state.db"))
    return replace(cfg, validator=v)


class FakeChain:
    """The chain client the weight loop talks to. Every weight push is logged
    with the block it was made at and the thread that made it."""

    def __init__(self, block: int, n: int = 256):
        self.block = int(block)
        self.hotkeys = [f"w19hk{i}" for i in range(n)]
        self.hotkeys[7] = KING
        self.pushes: list[tuple[int, str, int]] = []
        self._lock = threading.Lock()

    def current_block(self) -> int:
        return self.block

    def n_uids(self) -> int:
        return len(self.hotkeys)

    def registered_hotkeys(self) -> list[str]:
        return list(self.hotkeys)

    def uid_for_hotkey(self, hotkey):
        return self.hotkeys.index(hotkey) if hotkey in self.hotkeys else None

    def _log(self, kind: str) -> None:
        with self._lock:
            self.pushes.append((self.block, kind, threading.get_ident()))

    def set_equal_share_weights(self, reward_uids, n_uids, *, decay, burn_uid):
        self._log("equal")

    def set_weight_vector(self, weights):
        self._log("vector")
        return True

    def push_blocks(self) -> list[int]:
        return [b for b, _, _ in self.pushes]


class HungStep:
    """A lane D step that hangs until released (or :data:`HANG_S`): records
    each call's boundary, thread and daemon flag."""

    def __init__(self, *, raises: BaseException | None = None):
        self.release = threading.Event()
        self.started = threading.Event()
        self.calls: list[int] = []
        self.threads: list[int] = []
        self.daemon: list[bool] = []
        self.raises = raises

    def __call__(self, boundary):
        self.calls.append(int(boundary))
        self.threads.append(threading.get_ident())
        self.daemon.append(threading.current_thread().daemon)
        self.started.set()
        if self.raises is not None:
            raise self.raises
        self.release.wait(HANG_S)
        return {"boundary": int(boundary)}


class HungBenchHost:
    """A bench pod whose sweep hangs (a multi-hour lane D bench sweep)."""

    def __init__(self):
        self.release = threading.Event()
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def __call__(self, digests, suites):
        self.calls.append((tuple(digests), tuple(suites)))
        self.release.wait(HANG_S)
        return {}


def sweep_step(host: HungBenchHost):
    """A lane D step that runs the paired bench sweep (W3b2/W5e
    ``SweepSession``) for the settlement: the sweep is the slow part."""
    from cascade.validator.lane_d_bench import SweepSession

    def step(boundary):
        session = SweepSession(f"w19-{int(boundary)}", host)
        return session.paired("c" * 64, "k" * 64, "h" * 64, suites=("gift_eval",))
    return step


def runner(state_dir: Path, *, armed: bool, step=None):
    """A real ValidatorRunner (genesis champion state, king at uid 7) with
    ``lane_d_step`` set to ``step``. Asserts the W19 surface exists first so
    a missing surface fails as an assertion."""
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.state import genesis

    names = {f.name for f in fields(ValidatorRunner)}
    assert "lane_d_step" in names, "ValidatorRunner needs a `lane_d_step` field (W19)"
    assert callable(getattr(ValidatorRunner, "_lane_d_tick", None)), \
        "ValidatorRunner needs a `_lane_d_tick(client)` method (W19)"
    r = ValidatorRunner(cfg=chain_cfg(state_dir, armed=armed), state=genesis(KING, 7))
    r.lane_d_step = step
    return r


def baseline_runner(state_dir: Path, *, armed: bool):
    """Today's weight loop with no lane D step at all (no W19 surface used)."""
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.state import genesis

    return ValidatorRunner(cfg=chain_cfg(state_dir, armed=armed), state=genesis(KING, 7))


def drive(r, chain: FakeChain, blocks, *, lane_d: bool = True) -> list[float]:
    """The validator poll loop's lane D and weight steps, one poll per block
    in ``blocks``: ``_lane_d_tick`` then the head re-assert (run_forever's
    order). Returns each poll's wall-clock seconds."""
    spent = []
    for b in blocks:
        chain.block = int(b)
        t0 = time.monotonic()
        if lane_d:
            r._lane_d_tick(chain)
        r._reassert_weights_at_head(chain)
        spent.append(time.monotonic() - t0)
    return spent


def module_source(name: str) -> str:
    """The source of the module ``import name`` loads (the code under test,
    not a path guessed from this file)."""
    import importlib

    mod = importlib.import_module(name)
    return Path(mod.__file__).read_text(encoding="utf-8")


def module_source_static(name: str) -> str:
    """The source of module ``name`` located WITHOUT executing it (static
    scans must not load what they check for). Fails if it does not exist."""
    import importlib.util

    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin, f"module {name} not found (W19 PATHS)"
    return Path(spec.origin).read_text(encoding="utf-8")


def polls(start: int, stop: int) -> list[int]:
    return list(range(int(start), int(stop), POLL_BLOCKS))


def gaps(push_blocks: list[int]) -> list[int]:
    return [b - a for a, b in zip(push_blocks, push_blocks[1:], strict=False)]
