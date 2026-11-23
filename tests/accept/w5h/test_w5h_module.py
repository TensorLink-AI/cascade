"""W5h: where the queue-selection code lives (README.md "Where the code
lives"). W5g's `test_every_reason_the_settlement_can_emit_has_a_case` scans
`cascade/validator/lane_d.py` for reason literals and lists `coldkey_changed`
and `reveal_delay_exceeded` as owned by W5h; they must therefore be emitted
from `cascade/validator/lane_d_queue.py`, so W5g stays green (CROSS)."""

from __future__ import annotations

import ast

from .helpers import LANE_D_PY, QUEUE_PY, code_literals

W5H_REASONS = ("coldkey_changed", "reveal_delay_exceeded", "coldkey_cap_settlement",
               "coldkey_unresolved")


def test_the_queue_module_exists_and_imports_light():
    import cascade.validator.lane_d_queue as mod

    assert QUEUE_PY.is_file()
    tree = ast.parse(QUEUE_PY.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not imported & {"torch", "bittensor"}, f"heavy imports in lane_d_queue: {imported}"
    assert mod.__name__ == "cascade.validator.lane_d_queue"


def test_the_w5h_reasons_are_emitted_from_the_queue_module():
    assert QUEUE_PY.is_file(), f"{QUEUE_PY.relative_to(QUEUE_PY.parents[2])} does not exist"
    lits = code_literals(QUEUE_PY)
    missing = [r for r in W5H_REASONS if r not in lits]
    assert not missing, f"lane_d_queue.py emits none of {missing} (as non-docstring literals)"


def test_lane_d_py_holds_no_w5h_only_reason_literal():
    """Kept true on main and after W5h: W5g's scan of lane_d.py would fail on
    a `coldkey_changed` / `reveal_delay_exceeded` literal there."""
    lits = code_literals(LANE_D_PY)
    assert "coldkey_changed" not in lits and "reveal_delay_exceeded" not in lits


def test_settle_lane_d_uses_the_queue_module():
    tree = ast.parse(LANE_D_PY.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    assert "lane_d_queue" in names or "cascade.validator.lane_d_queue" in names, \
        f"lane_d.py does not import the queue module: {sorted(names)}"
