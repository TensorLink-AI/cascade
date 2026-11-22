"""W5d / T7: context-variant measurements and their evidence verifier
(plan rev6 §4 T7, §1.6, §17.1 L4). Contract: README.md.

``scripts/laned/accept W5d --evidence evidence/t7.json`` exports the file as
``$LANED_EVIDENCE``. The ``test_evidence_*`` tests verify it with
:func:`helpers.evidence_problems` (schema ``laned_t7/1``): every ``fits``,
``armed``, ``identical`` and ``passed`` is re-derived from the recorded
numbers, so a hand-edited file fails. Without ``$LANED_EVIDENCE`` they SKIP
with a clear reason, so W5d reaches L3 before the pod run; L4 needs the
``--evidence`` run green on main. When it IS set, missing, malformed, CI-made
or hand-edited evidence fails hard.

The ``test_builder_*`` / ``test_verifier_*`` tests run offline: they build
evidence with ``cascade.eval.lane_d_t7`` from given numbers (no model, no
torch) and prove the verifier accepts it and rejects every tampering.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from .helpers import (
    DECLARED_128K,
    HORIZON,
    LAUNCH_CONTEXTS,
    MAX_BUDGET_WALL_S,
    MIN_MODELS,
    PINNED_TORCH,
    S8_HOST,
    T7_SCHEMA,
    TRAINED_CONTEXT,
    WP,
    evidence_problems,
    pyramid_contract,
    series,
    tiny_checkpoint,
)

torch_mark = pytest.mark.torch

LAUNCH_HOST = "valpriv-4090"
GPU = "NVIDIA GeForce RTX 4090"
TOOL = {"git_sha": "1f" * 20, "git_dirty": False}
# Measured-looking numbers for the offline builder tests (inputs, not evidence).
WALL = {4096: 610.0, 8192: 1490.0, 16384: 4310.0}
MEM = {4096: 3.1, 8192: 7.4, 16384: 19.8}


def _t7():
    import cascade.eval.lane_d_t7 as t7

    return t7


def _rows(t7, wall=None, mem=None, variants=LAUNCH_CONTEXTS, host=LAUNCH_HOST):
    wall = {**WALL, **(wall or {})}
    mem = {**MEM, **(mem or {})}
    budget = t7.settlement_budget(24.0)
    return [t7.variant_row(host=host, gpu=GPU, variant=v, wall_s=wall[v], peak_mem_gb=mem[v],
                           budget=budget, n_models=MIN_MODELS, n_windows=2000)
            for v in variants]


def _short(max_abs_diff=0.0):
    return {"base_context": TRAINED_CONTEXT, "declared_context": DECLARED_128K,
            "n_windows": 400, "max_abs_diff": max_abs_diff}


def _good(**kw) -> dict:
    t7 = _t7()
    rows = kw.pop("rows", None) or _rows(t7)
    return t7.build_evidence(rows, short_window=kw.pop("short_window", _short()), tool=dict(TOOL),
                             torch_version=kw.pop("torch_version", PINNED_TORCH),
                             source=kw.pop("source", "pod"))


def _evidence() -> dict:
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE is unset: no T7 evidence to verify yet. L4 needs "
                    "`scripts/laned/accept W5d --evidence evidence/t7.json` green on main "
                    "(the T7 pod run writes evidence/t7.json with cascade.eval.lane_d_t7; "
                    "see tests/accept/w5d/README.md)")
    p = Path(path)
    assert p.is_file(), f"LANED_EVIDENCE={path} does not exist"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AssertionError(f"LANED_EVIDENCE={path} is not JSON: {e}") from None


# ── the pod's evidence ───────────────────────────────────────────────────────


def test_evidence_schema_and_rederived_numbers():
    problems = evidence_problems(_evidence())
    assert problems == [], "T7 evidence rejected:\n  " + "\n  ".join(problems)


def test_evidence_t7_gate_identical_short_windows_and_4096_fits():
    ev = _evidence()
    assert ev.get("short_window", {}).get("identical") is True, (
        "T7: a 4k model and its declared-128k copy must score identically on ≤ 4096-history "
        f"windows; got {ev.get('short_window')}")
    assert TRAINED_CONTEXT in ev.get("armed", []), ev.get("armed")
    assert ev.get("passed") is True, ev.get("passed")


# ── the builder (offline, no torch) ──────────────────────────────────────────


def test_builder_constants():
    t7 = _t7()
    assert t7.SCHEMA == T7_SCHEMA
    assert tuple(t7.LAUNCH_VARIANTS) == LAUNCH_CONTEXTS
    assert t7.S8_HOST == S8_HOST


def test_builder_budget_is_half_a_settlement():
    t7 = _t7()
    b = t7.settlement_budget(24.0)
    assert b == {"wall_s": MAX_BUDGET_WALL_S, "mem_gb": 24.0}
    assert t7.settlement_budget(96.0, settlement_blocks=450)["wall_s"] == MAX_BUDGET_WALL_S / 2


def test_builder_row_fits_only_inside_wall_and_memory():
    t7 = _t7()
    budget = t7.settlement_budget(24.0)

    def row(wall, mem):
        return t7.variant_row(host=LAUNCH_HOST, gpu=GPU, variant=8192, wall_s=wall,
                              peak_mem_gb=mem, budget=budget, n_models=MIN_MODELS, n_windows=10)

    r = row(1000.0, 10.0)
    assert r["fits"] is True and r["variant"] == 8192 and r["budget"] == budget
    assert row(MAX_BUDGET_WALL_S, 24.0)["fits"] is True
    assert row(MAX_BUDGET_WALL_S + 0.5, 10.0)["fits"] is False
    assert row(1000.0, 24.5)["fits"] is False


def test_builder_row_refuses_malformed_input():
    t7 = _t7()
    budget = t7.settlement_budget(24.0)
    good = {"host": LAUNCH_HOST, "gpu": GPU, "variant": 4096, "wall_s": 1.0, "peak_mem_gb": 1.0,
            "budget": budget, "n_models": MIN_MODELS, "n_windows": 1}
    for bad in ({"variant": 12288}, {"variant": 2048}, {"n_models": MIN_MODELS - 1},
                {"n_windows": 0}, {"wall_s": 0.0}, {"peak_mem_gb": float("nan")},
                {"host": ""}, {"variant": 32768}, {"variant": 8192, "host": S8_HOST}):
        with pytest.raises(ValueError):
            t7.variant_row(**{**good, **bad})
    t7.variant_row(**{**good, "variant": 32768, "host": S8_HOST})


def test_verifier_accepts_builder_evidence():
    ev = _good()
    assert ev["schema"] == T7_SCHEMA and ev["wp"] == WP and ev["source"] == "pod"
    assert [r["variant"] for r in ev["rows"]] == list(LAUNCH_CONTEXTS)
    assert ev["armed"] == list(LAUNCH_CONTEXTS)
    assert ev["short_window"]["identical"] is True
    assert ev["passed"] is True
    assert evidence_problems(ev, check_git=False) == []
    json.dumps(ev)


def test_builder_arms_only_variants_that_fit():
    t7 = _t7()
    ev = _good(rows=_rows(t7, wall={16384: MAX_BUDGET_WALL_S + 900.0}))
    assert ev["armed"] == [4096, 8192]
    assert [r["fits"] for r in ev["rows"]] == [True, True, False]
    assert ev["passed"] is True
    assert evidence_problems(ev, check_git=False) == []
    ev = _good(rows=_rows(t7, mem={8192: 30.0, 16384: 60.0}))
    assert ev["armed"] == [4096]


def test_builder_fails_when_short_windows_differ():
    ev = _good(short_window=_short(max_abs_diff=1.5e-7))
    assert ev["short_window"]["identical"] is False and ev["passed"] is False
    assert evidence_problems(ev, check_git=False) == []


def test_builder_fails_without_every_launch_variant_or_when_4096_does_not_fit():
    t7 = _t7()
    ev = _good(rows=_rows(t7, variants=(4096, 8192)))
    assert ev["passed"] is False
    assert any("launch variants" in p for p in evidence_problems(ev, check_git=False))
    ev = _good(rows=_rows(t7, wall={4096: MAX_BUDGET_WALL_S + 1.0}))
    assert ev["passed"] is False and 4096 not in ev["armed"]
    assert evidence_problems(ev, check_git=False) == []


def test_builder_with_s8_rows():
    t7 = _t7()
    budget = t7.settlement_budget(96.0)
    s8 = [t7.variant_row(host=S8_HOST, gpu="NVIDIA RTX PRO 6000", variant=v, wall_s=w,
                         peak_mem_gb=m, budget=budget, n_models=MIN_MODELS, n_windows=300)
          for v, w, m in ((32768, 3100.0, 41.0), (65536, 5300.0, 77.0), (131072, 9800.0, 140.0))]
    ev = _good(rows=_rows(t7) + s8)
    assert ev["armed"] == [4096, 8192, 16384, 32768, 65536]
    assert evidence_problems(ev, check_git=False) == []


# ── the verifier rejects tampering ───────────────────────────────────────────


def _tampered(edit) -> list[str]:
    ev = copy.deepcopy(_good())
    edit(ev)
    return evidence_problems(ev, check_git=False)


@pytest.mark.parametrize("edit, prefix", [
    (lambda e: e.__setitem__("schema", "laned_t7/0"), "schema"),
    (lambda e: e.__setitem__("wp", "W0b"), "wp"),
    (lambda e: e.__setitem__("source", "ci"), "source"),
    (lambda e: e.__setitem__("torch", "2.5.0+cu124"), "torch"),
    (lambda e: e["tool"].__setitem__("git_dirty", True), "tool.git_dirty"),
    (lambda e: e["tool"].__setitem__("git_sha", "main"), "tool.git_sha"),
    (lambda e: e.__setitem__("measured_at", "yesterday"), "measured_at"),
    (lambda e: e.__setitem__("rows", []), "rows"),
    (lambda e: e["rows"][2].__setitem__("fits", False), "rows[2].fits"),
    (lambda e: e["rows"][1].__setitem__("wall_s", MAX_BUDGET_WALL_S * 2), "rows[1].fits"),
    (lambda e: e["rows"][0]["budget"].__setitem__("wall_s", 10800.0), "rows[0].budget"),
    (lambda e: e["rows"][2].__setitem__("host", S8_HOST), "rows[2].host"),
    (lambda e: e["rows"][0].__setitem__("n_models", 1), "rows[0].n_models"),
    (lambda e: e["rows"][0].__setitem__("variant", 12288), "rows[0].variant"),
    (lambda e: e["rows"].__setitem__(1, copy.deepcopy(e["rows"][0])), "rows"),
    (lambda e: e.__setitem__("armed", [4096, 8192, 16384, 32768]), "armed"),
    (lambda e: e["short_window"].__setitem__("max_abs_diff", 2e-6), "short_window.identical"),
    (lambda e: e["short_window"].__setitem__("declared_context", 16384), "short_window"),
    (lambda e: e.__setitem__("passed", False), "passed"),
])
def test_verifier_rejects_a_hand_edit(edit, prefix):
    problems = _tampered(edit)
    assert any(p.startswith(prefix) for p in problems), problems


def test_verifier_rejects_a_hidden_failure():
    """A failing run edited to look passing: identical flipped, passed flipped."""
    ev = _good(short_window=_short(max_abs_diff=3e-5))
    assert ev["passed"] is False
    ev["short_window"]["identical"] = True
    ev["passed"] = True
    problems = evidence_problems(ev, check_git=False)
    assert any(p.startswith("short_window.identical") for p in problems), problems


def test_verifier_rejects_ci_made_evidence():
    ev = _good(source="ci")
    assert ev["source"] == "ci"
    assert any(p.startswith("source") for p in evidence_problems(ev, check_git=False))


def test_verifier_rejects_an_unknown_tool_sha():
    ev = _good()
    ev["tool"]["git_sha"] = "0123456789abcdef0123456789abcdef01234567"
    assert any("not a commit" in p for p in evidence_problems(ev, check_git=True))


# ── measurement helpers ──────────────────────────────────────────────────────


def test_short_window_check_refuses_history_over_4096_before_loading(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", None)
    t7 = _t7()
    with pytest.raises(ValueError):
        t7.short_window_check(tmp_path / "absent", [series(256, 0), series(4097, 1)],
                              horizon=HORIZON)


@torch_mark
def test_measure_variant_and_short_window_check_on_a_tiny_model(tmp_path):
    import torch

    t7 = _t7()
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        ckpt = tiny_checkpoint(tmp_path / "entry", pyramid_contract(), seed=11,
                               median_shift=0.0, bf16=True)
        hists = [series(n, seed=n) for n in (64, 900, 4096, 6000)]
        budget = t7.settlement_budget(24.0)
        row = t7.measure_variant(ckpt, hists, context=8192, horizon=HORIZON, host=LAUNCH_HOST,
                                 gpu="cpu", n_models=MIN_MODELS, budget=budget, device="cpu")
        sw = t7.short_window_check(ckpt, hists[:3], horizon=HORIZON)
    finally:
        torch.set_num_threads(threads)
    assert row["variant"] == 8192 and row["n_windows"] == len(hists)
    assert row["n_models"] == MIN_MODELS and row["budget"] == budget
    assert np.isfinite(row["wall_s"]) and row["wall_s"] > 0
    assert np.isfinite(row["peak_mem_gb"]) and row["peak_mem_gb"] > 0
    assert row["fits"] is (row["wall_s"] <= budget["wall_s"]
                           and row["peak_mem_gb"] <= budget["mem_gb"])
    assert sw["base_context"] == TRAINED_CONTEXT and sw["declared_context"] == DECLARED_128K
    assert sw["n_windows"] == 3 and sw["max_abs_diff"] == 0.0 and sw["identical"] is True
    ev = t7.build_evidence(_rows(t7), short_window=sw, tool=dict(TOOL),
                           torch_version=PINNED_TORCH, source="pod")
    assert evidence_problems(ev, check_git=False) == []
