"""Every custom ``torch.autograd.Function`` in cascade has a bf16-autocast
backward test (2026-11-27 RCA guard, docs/launch-lessons-2026-11.md).

2026-11-27 06:48: the first rc4 lane B leg died in its first backward,
``_ExpertMatmul.backward`` "expected scalar type Float but found BFloat16".
Training runs under bf16 autocast; #970's parity tests ran fp32 without
autocast, so a paid 8xH200 pod found the bug instead of CI (#974).

This is an AST scan, torch-free: every class in ``cascade/`` that subclasses
``Function`` / ``*.autograd.Function`` must be named by a test file under
``tests/`` that also runs ``torch.autocast(..., dtype=torch.bfloat16)`` and
calls ``.backward(``. A new Function without such a test fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _is_autograd_function(base: ast.expr) -> bool:
    if isinstance(base, ast.Attribute):
        return base.attr == "Function" and (
            isinstance(base.value, ast.Attribute) and base.value.attr == "autograd"
            or isinstance(base.value, ast.Name) and base.value.id == "autograd")
    return isinstance(base, ast.Name) and base.id == "Function"


def custom_functions(root: Path = ROOT) -> list[tuple[str, str]]:
    found = []
    for path in sorted((root / "cascade").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(map(_is_autograd_function, node.bases)):
                found.append((path.relative_to(root).as_posix(), node.name))
    return found


def bf16_backward_tests(name: str, root: Path = ROOT) -> list[str]:
    out = []
    for path in sorted((root / "tests").rglob("test_*.py")):
        if path.name == Path(__file__).name:     # this scanner quotes every keyword
            continue
        text = path.read_text(encoding="utf-8")
        if (name in text and "autocast(" in text and "bfloat16" in text
                and ".backward(" in text):
            out.append(path.relative_to(root).as_posix())
    return out


def test_the_scan_finds_the_expert_matmul():
    # the scan itself: the Function behind the 2026-11-27 crash is seen
    assert ("cascade/trainer/expert_training.py", "_ExpertMatmul") in custom_functions()


def test_every_custom_autograd_function_has_a_bf16_autocast_backward_test():
    missing = [f"{rel}::{name}" for rel, name in custom_functions()
               if not bf16_backward_tests(name)]
    assert not missing, (
        "custom autograd Function(s) without a bf16-autocast backward test (training runs "
        "under bf16 autocast; #974): " + ", ".join(missing)
        + ". Add a test that runs it under torch.autocast(..., dtype=torch.bfloat16) and "
        "calls .backward() (CPU autocast is enough for CI).")


def test_the_scan_flags_a_function_without_one(tmp_path):
    (tmp_path / "cascade").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "cascade" / "k.py").write_text(
        "import torch\nclass _Mm(torch.autograd.Function):\n    pass\n"
        "from torch.autograd import Function\nclass _Other(Function):\n    pass\n")
    (tmp_path / "tests" / "test_k.py").write_text(
        "def test_fp32_only():\n    _Mm.apply(x).sum().backward()\n")   # the #970 shape
    assert custom_functions(tmp_path) == [("cascade/k.py", "_Mm"), ("cascade/k.py", "_Other")]
    assert bf16_backward_tests("_Mm", tmp_path) == []
    (tmp_path / "tests" / "test_k_bf16.py").write_text(
        "def test_bf16():\n    with torch.autocast('cpu', dtype=torch.bfloat16):\n"
        "        out = _Mm.apply(x)\n    out.sum().backward()\n")
    assert bf16_backward_tests("_Mm", tmp_path) == ["tests/test_k_bf16.py"]
