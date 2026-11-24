"""WB26 trainer side: a settling suite-target leg carries the era anchor
king's public bench rows (``ExpertEntry.public_rows["anchor"]``), benched
once per anchor, and the self-judge is guarded against the same anchor — so
a suite target can actually win. The rolling trainer derives the anchor with
the validator's own rule (``lane_b_anchor_pointer``)."""

from __future__ import annotations

import inspect

from . import helpers as h


def _entry(sc):
    (em,) = sc.ops.published
    (e,) = em.entries
    return e


def test_suite_leg_carries_anchor_rows_and_self_judge_gets_the_anchor(tmp_path):
    sc = h.trainer_scenario(tmp_path)
    h.trainer_settle(sc, tmp_path, anchor_pointer=h.ANCHOR_PTR)
    e = _entry(sc)
    assert e.domain == "suite:gift"
    rows = e.public_rows or {}
    assert "anchor" in rows, f"a suite-target entry carries public_rows['anchor']; got {sorted(rows)}"
    assert rows["anchor"] == h.bench_rows({}), "the anchor rows are the anchor checkpoint's bench rows"
    assert h.ANCHOR_PTR in sc.ops.fetched
    (call,) = sc.ops.judge_calls
    assert call["kw"].get("anchor_pointer") == h.ANCHOR_PTR, call["kw"]


def test_domain_leg_never_benches_the_anchor(tmp_path):
    sc = h.trainer_scenario(tmp_path, target="energy")
    h.trainer_settle(sc, tmp_path, anchor_pointer=h.ANCHOR_PTR)
    e = _entry(sc)
    assert "anchor" not in (e.public_rows or {}), sorted(e.public_rows)
    assert h.ANCHOR_PTR not in sc.ops.fetched, sc.ops.fetched
    (call,) = sc.ops.judge_calls
    assert "anchor_pointer" not in call["kw"], call["kw"]


def test_anchor_bench_rows_are_cached_per_anchor(tmp_path):
    sc = h.trainer_scenario(tmp_path)
    h.trainer_settle(sc, tmp_path, anchor_pointer=h.ANCHOR_PTR)
    rows = sc.driver._anchor_bench_rows(h.ANCHOR_PTR, h.SUITES)
    assert rows == h.bench_rows({})
    anchor_dir = tmp_path / "anchor"
    assert sc.ops.benched.count(anchor_dir) == 1, sc.ops.benched


def test_rolling_derives_the_anchor_with_the_validator_rule():
    """The trainer passes ``lane_b_anchor_pointer(cur)`` — the rule the
    validator applies to the signed manifest built from the same era."""
    from cascade.trainer import rolling

    src = inspect.getsource(rolling.RollingScheduler._lane_b_settle)
    assert "lane_b_anchor_pointer" in src and "anchor_pointer" in src, (
        "RollingScheduler._lane_b_settle passes the era's anchor (lane_b_anchor_pointer) to "
        "LaneBDriver.settle")


def test_runner_ops_judge_forwards_the_anchor():
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    assert h.has_param(RunnerLaneBOps.judge, "anchor_pointer"), (
        "RunnerLaneBOps.judge forwards the anchor king to judge_manifest (WB26)")
    assert inspect.signature(RunnerLaneBOps.judge).parameters["anchor_pointer"].default is None
