"""Runtime box guard (retro P2): call-time heavy imports refused by name, the
torch-free finiteness check, the shared per-step RSS sampler, and every box
service installing the guard.

Torch-free: torch is never imported. The refusal tests use a fake heavy
package (``fakeheavy``) or assert that torch is REFUSED in a subprocess; the
allowed paths only ever import the fake one.
"""

from __future__ import annotations

import importlib
import json
import logging
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def _mod(name: str):
    """The module under test, or an assertion failure naming it (on a base
    without the box guard every test fails by assertion, not by import)."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise AssertionError(f"retro P2: {name} is missing ({e})") from e


def _bg():
    return _mod("cascade.shared.box_guard")


def _rb():
    return _mod("cascade.shared.rss_budget")


@pytest.fixture
def fake_heavy(tmp_path, monkeypatch):
    """A package ``fakeheavy`` (with a lazily imported submodule) treated as
    heavy, and a box module ``boxmod`` that imports it inside a function."""
    pkg = tmp_path / "fakeheavy"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "def lazy():\n    import fakeheavy.sub\n    return fakeheavy.sub.X\n")
    (pkg / "sub.py").write_text("X = 7\n")
    (tmp_path / "boxmod.py").write_text(
        "def call_time_import():\n    import fakeheavy\n    return fakeheavy\n\n"
        "def fallback():\n    try:\n        import fakeheavy  # noqa: F401\n"
        "        return 'heavy'\n    except ImportError:\n        return 'fallback'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("CASCADE_BOX_GUARD", raising=False)
    monkeypatch.delenv("CASCADE_BOX_GUARD_ALLOW", raising=False)
    try:
        bg = importlib.import_module("cascade.shared.box_guard")
    except ImportError:          # base: each test fails by assertion in _bg()
        bg = None
    if bg is not None:
        monkeypatch.setattr(bg, "HEAVY_PACKAGES", bg.HEAVY_PACKAGES | {"fakeheavy"})
    for m in ("fakeheavy", "fakeheavy.sub", "boxmod"):
        sys.modules.pop(m, None)
    yield
    if bg is not None:
        bg.uninstall()
    for m in ("fakeheavy", "fakeheavy.sub", "boxmod"):
        sys.modules.pop(m, None)


def test_call_time_import_is_refused_by_name_and_logged(fake_heavy, caplog):
    box_guard = _bg()
    assert box_guard.install("cascade-validator") == "enforce"
    boxmod = importlib.import_module("boxmod")
    with caplog.at_level(logging.ERROR, logger=box_guard.__name__), \
            pytest.raises(box_guard.BoxImportRefused) as ei:
        boxmod.call_time_import()
    msg = str(ei.value)
    assert "cascade-validator" in msg and "'fakeheavy'" in msg and "boxmod" in msg
    assert "boxmod.py:2" in msg, msg                     # the calling file:line
    assert isinstance(ei.value, ModuleNotFoundError)
    assert "fakeheavy" not in sys.modules
    errs = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errs and box_guard.LOG_TOKEN in errs[0]
    assert [r["caller"] for r in box_guard.refusals()] == ["boxmod"]


def test_import_module_path_is_refused_too(fake_heavy):
    box_guard = _bg()
    box_guard.install("cascade-intake")
    with pytest.raises(box_guard.BoxImportRefused):
        importlib.import_module("fakeheavy")
    # a find_spec probe (huggingface_hub-style availability check) never raises
    assert importlib.util.find_spec("fakeheavy") is not None


def test_already_loaded_package_is_still_refused_from_box_code(fake_heavy):
    box_guard = _bg()
    """A finder is never consulted for a cached module: the import-statement
    hook refuses it anyway (the window that loaded it does not leak)."""
    box_guard.install("cascade-validator")
    boxmod = importlib.import_module("boxmod")
    with box_guard.allow("fakeheavy", why="test window"):
        assert boxmod.call_time_import().__name__ == "fakeheavy"
    assert "fakeheavy" in sys.modules
    with pytest.raises(box_guard.BoxImportRefused):
        boxmod.call_time_import()


def test_the_package_internal_lazy_imports_are_never_refused(fake_heavy):
    box_guard = _bg()
    box_guard.install("cascade-validator")
    with box_guard.allow("fakeheavy", why="test window"):
        fh = importlib.import_module("fakeheavy")
    assert fh.lazy() == 7                        # fakeheavy importing fakeheavy.sub


def test_allow_window_is_thread_local_and_needs_a_reason(fake_heavy):
    box_guard = _bg()
    box_guard.install("cascade-validator")
    boxmod = importlib.import_module("boxmod")
    got = {}

    def other():
        try:
            boxmod.call_time_import()
            got["other"] = "imported"
        except box_guard.BoxImportRefused:
            got["other"] = "refused"

    with box_guard.allow("fakeheavy", why="test window"):
        t = threading.Thread(target=other)
        t.start()
        t.join(5)
    assert got == {"other": "refused"}
    with pytest.raises(ValueError), box_guard.allow("fakeheavy", why=""):
        pass


def test_except_importerror_fallbacks_still_fall_back(fake_heavy):
    box_guard = _bg()
    box_guard.install("cascade-validator")
    assert importlib.import_module("boxmod").fallback() == "fallback"


def test_warn_mode_logs_and_proceeds(fake_heavy, monkeypatch, caplog):
    box_guard = _bg()
    monkeypatch.setenv(box_guard.ENV_MODE, "warn")
    assert box_guard.install("cascade-trainer") == "warn"
    with caplog.at_level(logging.ERROR, logger=box_guard.__name__):
        assert importlib.import_module("boxmod").call_time_import().__name__ == "fakeheavy"
    assert any(box_guard.LOG_TOKEN in r.getMessage() for r in caplog.records)
    assert box_guard.refusals()[0]["mode"] == "warn"


def test_env_allowlist_and_off(fake_heavy, monkeypatch):
    box_guard = _bg()
    monkeypatch.setenv(box_guard.ENV_ALLOW, "fakeheavy")
    box_guard.install("cascade-validator")
    assert importlib.import_module("boxmod").call_time_import()
    box_guard.uninstall()
    sys.modules.pop("fakeheavy", None)
    monkeypatch.delenv(box_guard.ENV_ALLOW)
    monkeypatch.setenv(box_guard.ENV_MODE, "off")
    assert box_guard.install("cascade-validator") == "off"
    assert importlib.import_module("boxmod").call_time_import()
    assert box_guard.refusals() == []


GUARDED_TORCH = r"""
import json, sys
from cascade.shared import box_guard
box_guard.install("cascade-validator", default_mode="enforce")
out = {}
for mod in ("torch", "timestral", "transformers", "safetensors"):
    try:
        __import__(mod)
        out[mod] = "imported"
    except box_guard.BoxImportRefused as e:
        out[mod] = "refused" if "cascade-validator" in str(e) else str(e)
    except ImportError as e:
        out[mod] = "refused-or-absent"
out["loaded"] = sorted(m for m in ("torch", "timestral", "transformers", "safetensors")
                       if m in sys.modules)
print(json.dumps(out))
"""


def test_real_heavy_packages_are_refused_in_a_guarded_process():
    """The #1003 class: a box process importing timestral / torch at call
    time gets a named refusal, and nothing heavy is ever loaded."""
    proc = subprocess.run([sys.executable, "-P", "-c", GUARDED_TORCH], capture_output=True,
                          text=True, cwd=str(ROOT), timeout=120,
                          env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["loaded"] == [], got
    assert got["torch"] == "refused", got
    assert all(v in ("refused", "refused-or-absent") for k, v in got.items() if k != "loaded")


# ── every box service installs it ───────────────────────────────────────────

SERVICES = [
    ("cascade.validator.main", "off"),          # external validators: opt-in by env
    ("cascade.trainer.main", "warn"),
    ("cascade.funding.main", "enforce"),
    ("cascade.provision.main", "enforce"),
    ("cascade.lane_d.main", "enforce"),
    ("cascade.pool.pool_d_publish", "enforce"),
    ("cascade.pool.pool_d_spent", "enforce"),
    ("cascade.pool.settlement_evals", "enforce"),
]


def _install_calls(module: str) -> list[tuple[str, str]]:
    """``(service, default_mode)`` of each ``box_guard.install`` call in the
    module's ``main()`` (read from the source: running a service ``main``
    in a test would load env files and open chain access)."""
    import ast

    spec = importlib.util.find_spec(module)
    tree = ast.parse(Path(spec.origin).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    out = []
    for n in ast.walk(main):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "install" and getattr(n.func.value, "id", "") == "box_guard"):
            kw = {k.arg: k.value.value for k in n.keywords}
            out.append((n.args[0].value, kw.get("default_mode", "enforce")))
    return out


@pytest.mark.parametrize("module,default", SERVICES)
def test_every_box_service_main_installs_the_guard(module, default):
    calls = _install_calls(module)
    assert len(calls) == 1 and calls[0][1] == default, (module, calls)


def test_the_guard_is_installed_before_the_service_does_anything():
    """install() is the main's first statement after the release code-path
    check: nothing the service loads before it escapes the guard."""
    import ast

    _bg()

    for module, _ in SERVICES:
        spec = importlib.util.find_spec(module)
        tree = ast.parse(Path(spec.origin).read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        src = [ast.unparse(st) for st in main.body[:6]]
        firsts = [i for i, t in enumerate(src) if "box_guard.install" in t]
        assert firsts, f"{module}.main() does not install the box guard"
        first = firsts[0]
        before = " ".join(src[:first])
        for t in ("parse_args(", "load_env_files()", "basicConfig("):
            if module.endswith(("funding.main",)) and t in ("parse_args(", "basicConfig("):
                continue    # intake parses its own flags first (logging level)
            assert t not in before, (module, t, before)


# ── checkpoint guard: torch-free finiteness ─────────────────────────────────

def _write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, blobs, off = {}, [], 0
    for name, (dt, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hj)) + hj + b"".join(blobs))


def _bf16(values) -> np.ndarray:
    return (np.asarray(values, dtype="<f4").view("<u4") >> 16).astype("<u2")


def test_verify_weights_finite_is_torch_free_and_exact(tmp_path, fake_heavy):
    box_guard = _bg()
    from cascade.eval import checkpoint_guard as G

    box_guard.install("cascade-trainer")         # enforce: any torch/safetensors import raises
    d = tmp_path / "ck"
    d.mkdir()
    ok = {"a": ("F32", np.array([0.0, -0.0, 1e-45, 3.4e38], dtype="<f4")),
          "b": ("BF16", _bf16([1.0, -2.5, 0.0])),
          "c": ("F16", np.array([65504.0, -1.0], dtype="<f2")),
          "i": ("I64", np.array([-1, 2**62], dtype="<i8"))}
    _write_safetensors(d / G.WEIGHTS_FILE, ok)
    G.verify_weights_finite(d)
    bad = dict(ok)
    bad["a"] = ("F32", np.array([0.0, np.nan], dtype="<f4"))
    bad["b"] = ("BF16", _bf16([1.0, -np.inf]))
    _write_safetensors(d / G.WEIGHTS_FILE, bad)
    with pytest.raises(G.CheckpointDiverged, match="2 tensor"):
        G.verify_weights_finite(d)
    assert "torch" not in sys.modules and "safetensors" not in sys.modules
    assert box_guard.refusals() == []


def test_verify_weights_finite_refuses_offsets_outside_the_file(tmp_path):
    import inspect

    from cascade.eval import checkpoint_guard as G

    src = inspect.getsource(G.verify_weights_finite)
    assert "import torch" not in src and "safe_open" not in src, "finite check loads torch"

    d = tmp_path / "ck"
    d.mkdir()
    hj = json.dumps({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 1 << 20]}}).encode()
    (d / G.WEIGHTS_FILE).write_bytes(struct.pack("<Q", len(hj)) + hj + b"\0" * 16)
    with pytest.raises(ValueError):
        G.verify_weights_finite(d)


def test_the_state_spec_builder_runs_inside_a_named_torch_window(monkeypatch):
    box_guard = _bg()
    from cascade.eval import checkpoint_guard as G

    seen = []
    monkeypatch.setattr(G, "_model_for_contract_inner",
                        lambda c: seen.append(box_guard._allowed("torch")) or ("m", "k", {}))
    assert G._model_for_contract(object()) == ("m", "k", {})
    assert seen == [True]
    assert box_guard._allowed("torch") is False


# ── the shared per-step RSS sampler ─────────────────────────────────────────

def test_step_rss_logs_and_errors_over_budget_with_one_token(caplog):
    rss_budget = _rb()
    readings = iter([100 * 2**20] + [900 * 2**20] * 1000)
    lg = logging.getLogger("cascade.test.rss")
    with caplog.at_level(logging.INFO, logger=lg.name), \
            rss_budget.step_rss("unit step", 512 * 2**20, logger=lg, hint="go to a pod",
                                probe=lambda: next(readings), interval=0.01) as s:
        pass
    assert s.rss == {"start": 100 * 2**20, "peak": 900 * 2**20, "growth": 800 * 2**20}
    assert s.over is True
    msgs = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert any("unit step: rss start 100 MB, peak 900 MB (+800 MB)" in m for _l, m in msgs)
    err = [m for lv, m in msgs if lv >= logging.ERROR]
    assert err and err[0].startswith(rss_budget.LOG_TOKEN) and "512 MB box budget" in err[0]
    assert "go to a pod" in err[0]


def test_step_rss_reports_even_when_the_step_raises(caplog):
    rss_budget = _rb()
    lg = logging.getLogger("cascade.test.rss2")
    with caplog.at_level(logging.INFO, logger=lg.name), pytest.raises(RuntimeError), \
            rss_budget.step_rss("raising step", logger=lg):
        raise RuntimeError("boom")
    assert any("raising step: rss start" in r.getMessage() for r in caplog.records)


def test_decorator_form():
    rss_budget = _rb()
    calls = []

    @rss_budget.rss_budgeted("deco step", 1 << 40)
    def step(x):
        calls.append(x)
        return x * 2

    assert step(3) == 6 and calls == [3]


def test_offthread_steps_share_the_one_sampler():
    rss_budget = _rb()
    """#1000 and #1005 had their own samplers; both use rss_budget now."""
    from cascade.validator import marketcast, offthread_worker

    assert offthread_worker._RssWatch is rss_budget.RssWatch
    assert offthread_worker.process_rss_bytes is rss_budget.process_rss_bytes
    src = Path(marketcast.__file__).read_text()
    assert "_RssWatch" not in src and "rss_budget import" in src
    lb = (ROOT / "cascade/trainer/lane_b_driver.py").read_text()
    assert 'step_rss(f"lane B acceptance' in lb
