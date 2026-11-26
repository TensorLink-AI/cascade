"""WB25 by source (AST and text, never torch: these run on the box too).

An expert-slice leg gets its world size from ``gang_size`` like a lane A leg,
the selection reaches the Gang, helper ranks can build the same
``ExpertSlices`` leaves, and the lane A guards stay as they are. Contract:
README.md "Structure".
"""

from __future__ import annotations

import ast
import re

from . import helpers as h


def _train_pyramid() -> ast.FunctionDef:
    return h.function(h.module_tree(h.TRAINER), "train_pyramid")


def test_slice_legs_take_the_gang_size():
    """``world`` is ``gang_size(...)`` for every leg: no ``... if slices is
    None else 1`` override (main forces 1 GPU on expert legs)."""
    fn = _train_pyramid()
    values = h.assignments(fn, "world")
    assert values, "train_pyramid no longer assigns world"
    for v in values:
        assert not isinstance(v, ast.IfExp), \
            f"world is still a conditional expression: {ast.unparse(v)}"
        assert isinstance(v, ast.Call) and h.callee(v) == "gang_size", \
            f"world is not gang_size(...): {ast.unparse(v)}"
        assert "slices" not in h.names_in(v) and "expert_selection" not in h.names_in(v), \
            f"world still depends on the slices: {ast.unparse(v)}"
        arg_names = {ast.unparse(a) for a in v.args} | {ast.unparse(k.value) for k in v.keywords}
        assert "trainer.device" in arg_names, f"gang_size not called on trainer.device: {ast.unparse(v)}"


def test_no_world_one_override_anywhere_in_train_pyramid():
    src = ast.unparse(_train_pyramid())
    assert not re.search(r"if\s+slices\s+is\s+None\s+else\s+1\b", src), \
        "the `if slices is None else 1` override is still in train_pyramid"


def test_gang_receives_the_selection():
    """The Gang the leg builds is told the expert selection (``slices`` or
    ``expert_selection``), so its helper ranks train the same leaves."""
    gang_calls = h.calls(_train_pyramid(), "Gang")
    assert gang_calls, "train_pyramid builds no Gang"
    for call in gang_calls:
        used = set()
        for node in [*call.args, *(k.value for k in call.keywords)]:
            used |= h.names_in(node)
        assert used & {"slices", "expert_selection"}, \
            f"Gang(...) does not get the selection: {ast.unparse(call)}"


def test_helper_ranks_can_build_expert_slices():
    """pyramid_dp builds ``ExpertSlices`` (from ``.expert_training``) for a
    helper rank of an expert leg; main's helpers train lane A only."""
    tree = h.module_tree(h.DP)
    imported = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                and (n.module or "").endswith("expert_training")
                and any(a.name == "ExpertSlices" for a in n.names)]
    assert imported, "pyramid_dp never imports ExpertSlices from .expert_training"
    assert h.calls(tree, "ExpertSlices"), "pyramid_dp never constructs ExpertSlices"


def test_helper_ranks_accelerate_only_lane_a():
    """Rank 0 binds the fast kernels only when there are no slices, so every
    helper rank's ``accelerate(model)`` is conditional too (main calls it
    unconditionally: "the gang only trains lane A")."""
    tree = h.module_tree(h.DP)
    acc = h.calls(tree, "accelerate")
    assert acc, "pyramid_dp no longer calls accelerate (lane A helper ranks need it)"
    up = h.parents(tree)
    for call in acc:
        assert h.is_conditional(call, up, tree), \
            f"pyramid_dp line {call.lineno}: accelerate(...) is unconditional"


# ── lane A guards (pass on main; must stay) ─────────────────────────────────


def test_lane_a_rank0_still_accelerates_only_without_slices():
    fn = _train_pyramid()
    up = h.parents(fn)
    acc = h.calls(fn, "accelerate")
    assert acc, "train_pyramid no longer accelerates lane A"
    assert all(h.guarded_by_slices_none(c, up, fn) for c in acc), \
        "train_pyramid's accelerate(model) is no longer under `if slices is None:`"


def test_lane_a_keeps_the_gang_path():
    """Lane A still trains through pyramid_dp's Gang and micro_batch_pass."""
    fn = _train_pyramid()
    imports = {a.name for n in ast.walk(fn) if isinstance(n, ast.ImportFrom)
               and (n.module or "") == "pyramid_dp" for a in n.names}
    assert {"Gang", "gang_size", "micro_batch_pass", "owned_splits"} <= imports, imports
    assert h.calls(fn, "micro_batch_pass") and h.calls(fn, "shard_optimizer")


def test_gang_size_still_reads_cascade_train_gpus():
    fn = h.function(h.module_tree(h.DP), "gang_size")
    src = ast.unparse(fn)
    assert "GANG_ENV" in src and "device_count" in src
    assert "GANG_ENV = 'CASCADE_TRAIN_GPUS'" in ast.unparse(h.module_tree(h.DP))


def test_nccl_pin_unchanged():
    """The pinned all-reduce (bit-identical lane A legs across pods)."""
    tree = h.module_tree(h.DP)
    pin = next(n for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "NCCL_PIN" for t in n.targets))
    assert ast.literal_eval(pin.value) == {
        "NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple", "NCCL_NVLS_ENABLE": "0",
        "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"}


def test_torch_pin_unchanged():
    """The reproducibility contract: no dependency change."""
    assert '"torch==2.4.1"' in h.PYPROJECT.read_text(encoding="utf-8")
    lock = h.UV_LOCK.read_text(encoding="utf-8")
    assert re.search(r'^name = "torch"\nversion = "2\.4\.1(\+cu124)?"$', lock, re.M)


def test_main_snapshot_is_main():
    """The lane A oracle files are the cbe4c79 bytes (README)."""
    import hashlib

    for name, digest in h.SNAPSHOT_SHA256.items():
        raw = (h.HERE / h.SNAPSHOT_DIR / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, name


def test_plan_row():
    row = h.plan_row("WB25")
    assert row is not None, "no WB25 row in docs/plans/LANE_D_PLAN.md (plan rev6.8)"
    assert len(row) == 6, row
    assert "rev6.8" in row[1] and "gpu_n" in row[1] and "CASCADE_TRAIN_GPUS" in row[1]
    assert {d.strip() for d in row[3].split(",")} == {"WB16", "WB19"}
    assert row[4] == "ci + pod"
    assert row[5] == "`scripts/laned/accept WB25`"
