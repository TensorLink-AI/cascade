"""W8b in the validator loop: the production step runs off the poll/weight
thread through W19's seam, the bench sweep never delays `set_weights`, the
step reads the chain through its own reader (never the poll thread's
client), and run_forever wires it. Contract: README.md."""

from __future__ import annotations

import ast
import threading

import pytest

from . import helpers as h


@pytest.fixture
def cleanup():
    """Release every hung fixture at the end, so no thread outlives a test."""
    hung = []
    yield hung.append
    for obj in hung:
        obj.release.set()


SEG1 = h.polls(h.B1 - 300, h.B1 + 10)          # up to and including the B1 tick
SEG2 = h.polls(h.B1 + 10, h.B2 + 900)          # B2's gate sweep hangs across it
SEG3 = h.polls(h.B2 + 900, h.B2 + 1500)        # after the sweep returns


def test_inv6_hung_bench_sweep_never_delays_set_weights(tmp_path, cleanup):
    h.w19_surface()
    cfg = h.chain_cfg(tmp_path / "v")
    fx = h.fixture(tmp_path)
    host = h.BenchHost(fx["house"]["digest"], hang=True)
    cleanup(host)
    rig = h.Rig(tmp_path / "rig", cfg, fx, host=host)
    r = h.runner(cfg, step=rig.step)
    chain = h.PollChain(SEG1[0])
    main = threading.get_ident()

    spent = h.drive(r, chain, SEG1)
    assert r._lane_d_worker.join(30), "the B1 settlement never finished"
    assert rig.step.court_for(h.B1) == [None]
    spent += h.drive(r, chain, SEG2)
    assert host.started.wait(5), "the B2 gate sweep never started"
    assert r._lane_d_worker.running, "fixture: the sweep must still be hung"
    assert all(vec[h.CHALLENGER_UID] == 0.0 for _, vec, _ in chain.pushes), \
        "X was paid before its settlement was published"
    host.release.set()
    assert r._lane_d_worker.join(30), "the B2 settlement never finished after the sweep"
    spent += h.drive(r, chain, SEG3)

    base = h.runner(cfg)
    bchain = h.PollChain(SEG1[0])
    h.drive(base, bchain, SEG1 + SEG2 + SEG3)
    assert chain.push_blocks() == bchain.push_blocks(), "lane D moved the weight-set timing"
    assert max(spent) < h.TICK_BUDGET_S, f"a poll waited {max(spent):.2f}s on lane D"
    gaps = [b - a for a, b in zip(chain.push_blocks(), chain.push_blocks()[1:], strict=False)]
    assert max(gaps) <= h.INTERVAL + h.POLL_BLOCKS - 1
    assert chain.last_vector()[h.CHALLENGER_UID] == pytest.approx(0.25, rel=1e-12)
    assert {t for _, _, t in chain.pushes} == {main}, "weights pushed off the poll thread"
    assert host.threads and main not in host.threads, "the bench ran on the poll thread"
    assert rig.factory.threads and main not in rig.factory.threads
    assert all(t == main for t in chain.hash_threads), \
        "the lane D step used the poll thread's chain client"
    assert rig.published_report(h.B2)["crowned"] == fx["x"]


def test_chain_reader_is_built_on_the_step_thread(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v")
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))
    assert rig.factory.threads == [], "the reader was built on the constructing thread"
    seen: list[int] = []

    def run():
        seen.append(threading.get_ident())
        rig.step(h.B1)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(60)
    assert not t.is_alive()
    assert rig.factory.threads and set(rig.factory.threads) == set(seen)
    assert {th for _, th in rig.factory.reader.calls} == set(seen)


# ── run_forever wires the production step ──────────────────────────────────


def _run_forever() -> ast.FunctionDef:
    src = h.module_source("cascade.validator.loop")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "run_forever":
            return node
    raise AssertionError("ValidatorRunner.run_forever not found in cascade/validator/loop.py")


def _named(call: ast.Call, name: str) -> bool:
    f = call.func
    return (isinstance(f, ast.Name) and f.id == name) or \
        (isinstance(f, ast.Attribute) and f.attr == name)


def test_run_forever_builds_and_wires_the_lane_d_step():
    fn = _run_forever()
    builds = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
              and _named(n, "build_lane_d_step")]
    assert builds, "run_forever must build the lane D step (lane_d_wiring.build_lane_d_step)"
    assigned = [n for n in ast.walk(fn) if isinstance(n, ast.Assign | ast.AnnAssign)
                and any(isinstance(t, ast.Attribute) and t.attr == "lane_d_step"
                        for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    assert assigned, "run_forever must set self.lane_d_step"
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.While)]
    assert loops and all(b.lineno < loops[-1].lineno for b in builds), \
        "the step is wired before the poll loop starts"
    for call in builds:
        kw = {k.arg: k.value for k in call.keywords}
        assert "chain_factory" in kw, "build_lane_d_step needs chain_factory="
        v = kw["chain_factory"]
        poll_client = (isinstance(v, ast.Name) and v.id == "client") or (
            isinstance(v, ast.Lambda) and isinstance(v.body, ast.Name) and v.body.id == "client")
        assert not poll_client, "the lane D step must not share the poll thread's chain client"
