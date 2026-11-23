"""W8c wiring: unarmed nothing is built or read (invariant 1); the source
runs on its caller's thread (W19's worker) with its own chain reader built
there; nothing reachable from cascade.audit imports it (invariant 12); and
cascade-validator's main() passes it to build_lane_d_step as ``inputs`` and
its bench factory as ``bench``. Contract: README.md."""

from __future__ import annotations

import ast
import importlib.util
import threading

import pytest

from . import helpers as h


def test_inv1_unarmed_builds_nothing_and_reads_nothing(tmp_path):
    cfg = h.w8b.chain_cfg(tmp_path / "v", armed=False)
    state_dir, intake_dir = tmp_path / "never_state", tmp_path / "never_intake"
    out = h.inputs_mod().build_lane_d_inputs(
        cfg, state_dir=state_dir, intake_state_dir=intake_dir,
        chain_factory=h.Untouchable("chain_factory"), pool=h.Untouchable("pool"),
        eval_host=h.Untouchable("eval_host"), house_pointer=tmp_path / "never_house.json",
        house_signer_ss58=h.house_key().ss58_address,
        bench_host=h.Untouchable("bench_host"), expected_identity=h.identity())
    assert out is None, "[lane_d] from_block = 0: no source"
    assert not state_dir.exists() and not intake_dir.exists()


def test_inv1_armed_build_is_lazy(tmp_path):
    """Building the source (on the constructing thread) reads nothing: the
    first read happens at the first armed call, on the caller's thread."""
    rig = h.Rig(tmp_path)
    assert rig.factory.threads == [] and rig.pool.calls == []
    assert rig.eval_host.hellos == 0 and rig.eval_host.calls == []


def test_source_at_an_inactive_boundary_touches_nothing(tmp_path):
    rig = h.Rig(tmp_path)
    with pytest.raises(Exception) as ei:
        rig.source(h.ARM_BLOCK)          # from_block itself belongs to the era before
    h.assert_inputs_error(ei.value, "not_active")
    assert rig.factory.threads == [] and rig.pool.calls == [] and rig.eval_host.calls == []


def test_source_runs_on_the_callers_thread_with_its_own_reader(tmp_path):
    """Called from the W19 worker's thread, every chain read and eval host
    request happens on that thread; the reader is built there, not at build."""
    rig = h.Rig(tmp_path)
    main = threading.get_ident()
    seen: dict = {}

    def run():
        seen["tid"] = threading.get_ident()
        seen["inputs"] = rig.source(h.B1)

    t = threading.Thread(target=run)
    t.start()
    t.join(120)
    assert "inputs" in seen, "the source did not return on the worker thread"
    tid = seen["tid"]
    assert rig.factory.threads and set(rig.factory.threads) == {tid}
    assert main not in rig.factory.threads
    assert {t for _, t in rig.reader.calls} == {tid}
    assert {c["thread"] for c in rig.eval_host.calls} == {tid}


def test_source_through_the_w19_worker(tmp_path):
    """W8b's step over the source, run by W19's LaneDWorker off this thread,
    settles B1 (no exception reaches the worker)."""
    from cascade.validator.offthread_worker import LaneDWorker

    rig = h.Rig(tmp_path)
    store = h.w8b.Store()
    worker = LaneDWorker(rig.step(store=store))
    assert worker.submit(h.B1)
    assert worker.join(300)
    assert worker.last_error is None, f"the step failed: {worker.last_error!r}"
    from cascade.validator.lane_d_wiring import report_key

    assert report_key(h.B1) in store.objects
    assert threading.get_ident() not in rig.factory.threads


def test_inv12_audit_import_closure_reaches_no_w8c_module():
    from tests.accept.w9 import helpers as w9
    from tests.accept.w9.test_w9_independence import _closure

    for mod in (h.INPUTS, h.FORECASTS):
        assert importlib.util.find_spec(mod) is not None, f"{mod} does not exist"
    closure = _closure(["cascade.audit.lane_d", "cascade.audit.checks"])
    assert "cascade.validator.weights" in closure and "cascade.validator.loop" in closure
    assert h.INPUTS not in closure, "the audit's import closure reaches the W8c source"
    assert h.FORECASTS not in closure, "the audit's import closure reaches the eval-host side"
    bad = sorted(m for m in closure if w9.forbidden(m) and m != "torch")
    assert not bad, f"the audit's static import closure reaches {bad}"


def test_inv12_nothing_on_the_weight_path_imports_the_source():
    """loop.py and weights.py (in the audit's closure) never import it, even
    lazily; the entry point is the only caller."""
    from tests.accept.w9.test_w9_independence import _direct_imports

    for mod in ("cascade.validator.loop", "cascade.validator.weights",
                "cascade.validator.lane_d_wiring"):
        imps = _direct_imports(mod)
        assert h.INPUTS not in imps and h.FORECASTS not in imps, f"{mod} imports W8c"


def _validator_main():
    src = (h.w7a.REPO / "cascade" / "validator" / "main.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("main() not found in cascade/validator/main.py")


def _named(call: ast.Call, name: str) -> bool:
    f = call.func
    return (isinstance(f, ast.Name) and f.id == name) or \
        (isinstance(f, ast.Attribute) and f.attr == name)


def _is_none(v) -> bool:
    return isinstance(v, ast.Constant) and v.value is None


def test_validator_main_wires_the_source_into_the_step():
    fn = _validator_main()
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    sources = [n for n in calls if _named(n, "build_lane_d_inputs")]
    assert sources, "main() must build the W8c source (lane_d_inputs.build_lane_d_inputs)"
    steps = [n for n in calls if _named(n, "build_lane_d_step")]
    assert steps, "fixture: main() builds the W8b step"
    for call in steps:
        kw = {k.arg: k.value for k in call.keywords}
        assert "inputs" in kw and not _is_none(kw["inputs"]), \
            "build_lane_d_step needs inputs= (the W8c source)"
        assert "bench" in kw and not _is_none(kw["bench"]), \
            "build_lane_d_step needs bench= (the source's bench factory)"
    assert min(n.lineno for n in sources) < min(n.lineno for n in steps), \
        "the source is built before the step"
    for call in sources:
        kw = {k.arg: k.value for k in call.keywords}
        assert "chain_factory" in kw, "build_lane_d_inputs needs its own chain_factory="
        v = kw["chain_factory"]
        poll_client = (isinstance(v, ast.Name) and v.id == "client") or (
            isinstance(v, ast.Lambda) and isinstance(v.body, ast.Name) and v.body.id == "client")
        assert not poll_client, "the source must not share the poll thread's chain client"
