"""W19: lane D work runs off the weight/poll thread (invariant 6, T15).
A fixture slow lane D step leaves weight-set timing unchanged.
Contract: README.md."""

from __future__ import annotations

import ast
import threading

import pytest

from . import helpers as h

# polls from before the first armed boundary to past the second one: the
# step for B1 starts at B1 and (hung) spans B2 and well over 2 h of blocks
SPAN = h.polls(h.B1 - 300, h.B2 + 900)


@pytest.fixture
def cleanup():
    """Release every hung fixture at the end, so no thread outlives a test."""
    hung = []
    yield hung.append
    for obj in hung:
        obj.release.set()


def _baseline(tmp_path, *, armed):
    chain = h.FakeChain(SPAN[0])
    h.drive(h.baseline_runner(tmp_path / "baseline", armed=armed), chain, SPAN, lane_d=False)
    return chain.push_blocks()


# ── invariant 6: a hung lane D step never delays a due weight set ───────────


def test_inv6_hung_lane_d_step_leaves_weight_set_timing_unchanged(tmp_path, cleanup):
    want = _baseline(tmp_path, armed=True)
    step = h.HungStep()
    cleanup(step)
    r = h.runner(tmp_path / "w19", armed=True, step=step)
    chain = h.FakeChain(SPAN[0])
    spent = h.drive(r, chain, SPAN)
    assert step.started.wait(5), "the armed settlement at B1 never started its lane D step"
    assert chain.push_blocks() == want
    assert max(spent) < h.TICK_BUDGET_S, f"a poll waited {max(spent):.2f}s on lane D"
    assert step.calls == [h.B1]
    assert not step.release.is_set()          # still hung when the span ends


def test_inv6_weight_pushes_happen_on_the_poll_thread_not_the_lane_d_thread(tmp_path, cleanup):
    step = h.HungStep()
    cleanup(step)
    r = h.runner(tmp_path / "w19", armed=True, step=step)
    chain = h.FakeChain(SPAN[0])
    h.drive(r, chain, SPAN)
    assert step.started.wait(5)
    main = threading.get_ident()
    assert {t for _, _, t in chain.pushes} == {main}
    assert step.threads and step.threads[0] != main


def test_inv6_t15_two_hour_lane_d_sweep_leaves_no_weight_gap_over_the_interval(tmp_path, cleanup):
    """T15: "2 h lane D sweep across weight intervals: no weight gap >
    weight_set_interval_blocks" (on the poll grid: a push is due the first
    poll at or after last + interval)."""
    host = h.HungBenchHost()
    cleanup(host)
    r = h.runner(tmp_path / "w19", armed=True, step=h.sweep_step(host))
    chain = h.FakeChain(SPAN[0])
    spent = h.drive(r, chain, SPAN)
    pushed = chain.push_blocks()
    assert host.calls, "the lane D bench sweep never started"
    swept_from = h.B1
    during = [b for b in pushed if b >= swept_from]
    assert (during[-1] - swept_from) >= 600, "the sweep must span 2 h of blocks"
    assert len(during) >= 600 // h.INTERVAL
    assert max(h.gaps(pushed)) <= h.INTERVAL + h.POLL_BLOCKS - 1
    assert max(spent) < h.TICK_BUDGET_S
    assert pushed == _baseline(tmp_path, armed=True)


def test_inv6_raising_lane_d_step_never_reaches_the_weight_loop(tmp_path):
    step = h.HungStep(raises=RuntimeError("w19 fixture: settlement input unreadable"))
    r = h.runner(tmp_path / "w19", armed=True, step=step)
    chain = h.FakeChain(SPAN[0])
    h.drive(r, chain, SPAN)
    assert step.calls and step.calls[0] == h.B1
    assert chain.push_blocks() == _baseline(tmp_path, armed=True)


def test_one_lane_d_step_at_a_time_and_the_next_boundary_after_it_ends(tmp_path, cleanup):
    """While B1's step runs, later polls start nothing (no pile-up of
    threads); once it has ended, the next armed boundary's step starts."""
    step = h.HungStep()
    cleanup(step)
    r = h.runner(tmp_path / "w19", armed=True, step=step)
    chain = h.FakeChain(h.B1)
    h.drive(r, chain, h.polls(h.B1, h.B2 + 50))
    assert step.started.wait(5)
    assert step.calls == [h.B1]
    step.release.set()
    worker = r._lane_d_worker
    assert worker.join(10), "the released step did not finish"
    h.drive(r, chain, h.polls(h.B2 + 50, h.B2 + 100))
    assert step.calls[:2] == [h.B1, h.B2]


# ── invariant 1 / 10: unarmed is today's loop ───────────────────────────────


def test_weight_timing_unchanged_when_unarmed(tmp_path, cleanup):
    """[lane_d] from_block = 0: the step is never called, no lane D thread
    starts, and weight-set timing is today's."""
    want = _baseline(tmp_path, armed=False)
    step = h.HungStep()
    cleanup(step)
    r = h.runner(tmp_path / "w19", armed=False, step=step)
    chain = h.FakeChain(SPAN[0])
    before = threading.active_count()
    h.drive(r, chain, SPAN)
    assert step.calls == []
    assert threading.active_count() == before
    assert chain.push_blocks() == want


def test_weight_timing_armed_nonempty_lane_d_effect(tmp_path, cleanup):
    """Armed: the lane D step does run (once per armed boundary, off the poll
    thread), and weight-set timing is still today's."""
    want = _baseline(tmp_path, armed=True)
    step = h.HungStep()
    cleanup(step)
    r = h.runner(tmp_path / "w19", armed=True, step=step)
    chain = h.FakeChain(SPAN[0])
    h.drive(r, chain, SPAN)
    assert step.started.wait(5)
    assert step.calls == [h.B1]
    assert step.threads[0] != threading.get_ident()
    assert chain.push_blocks() == want


def test_no_step_wired_is_a_no_op(tmp_path):
    """Armed but no lane D step wired (W8b not landed): the tick does
    nothing and weights are today's."""
    r = h.runner(tmp_path / "w19", armed=True, step=None)
    chain = h.FakeChain(SPAN[0])
    h.drive(r, chain, SPAN)
    assert chain.push_blocks() == _baseline(tmp_path, armed=True)


def test_before_the_first_armed_boundary_nothing_starts(tmp_path, cleanup):
    """The settlement AT from_block belongs to the era before it
    (LaneDConfig.active): polls in [from_block, B1) start nothing."""
    step = h.HungStep()
    cleanup(step)
    r = h.runner(tmp_path / "w19", armed=True, step=step)
    chain = h.FakeChain(h.ARM_BLOCK)
    h.drive(r, chain, h.polls(h.ARM_BLOCK, h.B1))
    assert step.calls == []


def test_step_runs_on_a_daemon_thread(tmp_path, cleanup):
    """A hung step can never keep the validator process alive on shutdown."""
    step = h.HungStep()
    cleanup(step)
    r = h.runner(tmp_path / "w19", armed=True, step=step)
    chain = h.FakeChain(h.B1)
    h.drive(r, chain, [h.B1])
    assert step.started.wait(5)
    assert step.daemon == [True]


# ── the service loop calls the tick every poll ──────────────────────────────


def _run_forever_loop() -> ast.While:
    src = h.module_source("cascade.validator.loop")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "run_forever":
            loops = [n for n in ast.walk(node) if isinstance(n, ast.While)]
            assert loops, "run_forever has no poll loop"
            return loops[-1]
    raise AssertionError("ValidatorRunner.run_forever not found in cascade/validator/loop.py")


def _calls(node: ast.AST) -> list[str]:
    return [n.func.attr for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]


def test_run_forever_calls_the_lane_d_tick_every_poll():
    loop = _run_forever_loop()
    called = _calls(loop)
    assert "_lane_d_tick" in called, "run_forever's poll loop must call self._lane_d_tick(client)"
    assert "_reassert_weights_at_head" in called
