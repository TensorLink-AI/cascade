"""W19: "the weight-loop keep-alive (#471) is in the release". It exists on
main (``WeightsMixin._weights_keepalive_tick``, wired as the bench sweep's
``on_wait``); these tests keep it there. Contract: README.md."""

from __future__ import annotations

import ast

from . import helpers as h


def test_keepalive_tick_is_part_of_the_validator():
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.weights import WeightsMixin

    assert callable(getattr(WeightsMixin, "_weights_keepalive_tick", None))
    assert issubclass(ValidatorRunner, WeightsMixin)


def test_keepalive_re_pushes_the_last_vector_once_the_interval_passes(tmp_path):
    r = h.baseline_runner(tmp_path / "ka", armed=False)
    chain = h.FakeChain(h.B1)
    vec = r._apply_weights(chain, "w19-round", [7])
    assert vec and r._last_weight_block == h.B1
    chain.block = h.B1 + h.INTERVAL - 1
    r._weights_keepalive_tick()
    assert len(chain.pushes) == 1
    chain.block = h.B1 + h.INTERVAL
    r._weights_keepalive_tick()
    assert [b for b, kind, _ in chain.pushes] == [h.B1, h.B1 + h.INTERVAL]
    assert chain.pushes[-1][1] == "vector"
    assert r._last_weight_block == h.B1 + h.INTERVAL


def test_keepalive_never_raises_before_the_first_push(tmp_path):
    r = h.baseline_runner(tmp_path / "ka", armed=False)
    r._weights_keepalive_tick()


def test_bench_sweep_waits_tick_the_keepalive():
    """validator_bench passes the keep-alive as ``on_wait`` to the offloaded
    bench sweep (cascade/validator/validator_bench.py, #471)."""
    src = h.module_source("cascade.validator.validator_bench")
    wired = False
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.keyword) and node.arg == "on_wait":
            wired = wired or "_weights_keepalive_tick" in ast.unparse(node.value)
    assert wired
