"""W7d import-graph check (plan rev6 §3.0 W7d Accept "import-graph check
included"; §3.2 invariant 12: "Audit and satellite-replay code never import
the producer's eligibility, verdict or weight functions.").

Three independent views, all required (README "Invariant 12"):

1. static, direct: every module under `cascade/satellite/` imports, at any
   depth (function-level imports included), only the stdlib, numpy, its own
   package and the allowed shared primitives;
2. static, transitive: the closure over repo modules from those files holds
   no forbidden module, and nothing imports dynamically (the static graph
   could not follow it);
3. runtime: a fresh interpreter that imports the module, and one that runs a
   full replay, loads no forbidden module (lazy imports included) and no
   torch.

On main `cascade/satellite/` does not exist, so every test fails on an
assertion (or `ModuleNotFoundError: cascade.satellite`, in PATHS).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from .helpers import (
    ALLOWED_PACKAGES,
    ALLOWED_PRIMITIVES,
    CHAIN_TIMEOUT,
    MODULE,
    MODULE_PATH,
    REPLAY_SCRIPT,
    REPO,
    chain_house_bar,
    direct_imports,
    dynamic_imports,
    forbidden,
    module_name,
    satellite_modules,
    save_publication,
    static_closure,
)

_STDLIB = set(sys.stdlib_module_names)


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env.pop("PYTHONHASHSEED", None)
    return env


def test_the_replay_module_exists_at_its_pinned_path():
    assert MODULE_PATH.is_file(), f"{MODULE_PATH.relative_to(REPO)} does not exist"
    assert module_name(MODULE_PATH) == MODULE
    assert (REPO / "cascade" / "satellite" / "__init__.py").is_file(), \
        "cascade/satellite/ is not a package"


def test_inv12_satellite_direct_imports_are_only_stdlib_numpy_and_allowed_primitives():
    files = satellite_modules()
    assert MODULE_PATH in files, "no cascade/satellite/lane_d_replay.py to check"
    bad = {}
    for f in files:
        for imp in sorted(direct_imports(module_name(f))):
            top = imp.split(".")[0]
            if top in _STDLIB or top == "__future__" or top == "numpy":
                continue
            if imp.startswith("cascade.satellite") or imp in ALLOWED_PACKAGES:
                continue
            if any(imp == p or imp.startswith(p + ".") for p in ALLOWED_PRIMITIVES):
                continue
            bad.setdefault(f.name, []).append(imp)
    assert not bad, f"imports outside the invariant 12 allowlist: {bad}"


def test_inv12_satellite_static_closure_reaches_no_producer_module():
    files = satellite_modules()
    assert MODULE_PATH in files, "no cascade/satellite/lane_d_replay.py to check"
    closure = static_closure([module_name(f) for f in files])
    assert MODULE in closure
    bad = sorted(m for m in closure if forbidden(m))
    assert not bad, f"the satellite's static import closure reaches {bad}"


def test_inv12_satellite_imports_nothing_dynamically():
    files = satellite_modules()
    assert MODULE_PATH in files, "no cascade/satellite/lane_d_replay.py to check"
    found = [d for f in files for d in dynamic_imports(f)]
    assert not found, f"dynamic imports the static graph cannot check: {found}"


def test_inv12_importing_the_replay_loads_no_producer_module_and_no_torch():
    code = ("import sys, json; import " + MODULE + "; "
            "print(json.dumps(sorted(m for m in sys.modules "
            "if m.split('.')[0] in ('cascade', 'torch', 'bittensor'))))")
    out = subprocess.run([sys.executable, "-P", "-c", code], cwd=str(REPO), env=_env(),
                         capture_output=True, text=True, timeout=CHAIN_TIMEOUT)
    assert out.returncode == 0, out.stderr[-2000:]
    mods = json.loads(out.stdout.strip().splitlines()[-1])
    assert MODULE in mods
    bad = [m for m in mods if forbidden(m)]
    assert not bad, f"importing the replay loads {bad}"
    assert not [m for m in mods if m.split(".")[0] == "bittensor"], \
        "the SDK is imported at import time (only a signature check may load it)"


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = tmp_path_factory.mktemp("w7d-graph")
    c = chain_house_bar(root / "chain")
    return save_publication(c.pub, root / "pub")


def test_inv12_a_full_replay_loads_no_producer_module_and_no_torch(saved):
    """Lazy imports count: after replaying a whole chain (signatures, the
    bootstrap, the gate, pay and states) the interpreter holds no forbidden
    module."""
    out = subprocess.run([sys.executable, "-P", "-c", REPLAY_SCRIPT, str(saved)],
                         cwd=str(REPO), env=_env(), capture_output=True, text=True,
                         timeout=CHAIN_TIMEOUT)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert len(got["results"]) == 4, got["results"]
    bad = [m for m in got["modules"] if forbidden(m)]
    assert not bad, f"a replay loads {bad}"
