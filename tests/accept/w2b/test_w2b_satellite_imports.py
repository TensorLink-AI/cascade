"""W2b Accept: the satellite code path imports nothing from the module that
defines ``[lane_d_ops]`` (static import graph from cascade/validator/satellite.py),
and the satellite never reads the table.

The code path is satellite.py's own imports (module level and inside its
functions), their parent packages and the load-time import closure of all
of them. ``cascade/validator/__init__.py`` loads ``cascade.validator.loop``,
so the main validator may read the table only through a function-local
import, never a module-level one."""

from __future__ import annotations

from .helpers import (
    OPS_MODULE,
    SATELLITE_MODULE,
    module_file,
    satellite_code_path,
    string_constants,
)


def test_w2b_ops_module_exists():
    path, _ = module_file(OPS_MODULE)
    assert path is not None, f"{OPS_MODULE} does not exist"


def test_w2b_satellite_code_path_never_imports_the_ops_module():
    path, _ = module_file(OPS_MODULE)
    assert path is not None, f"{OPS_MODULE} does not exist (nothing to check against)"
    reached = satellite_code_path()
    # the walker really follows the satellite's imports
    for must in ("cascade.shared.config.loader", "cascade.shared.netconfig",
                 "cascade.shared.config.satellite", "cascade.shared.receipt"):
        assert must in reached, f"import walker missed {must}"
    assert OPS_MODULE not in reached, (
        f"the satellite code path imports {OPS_MODULE}")
    dynamic = sorted(m for m in reached if OPS_MODULE in string_constants(m))
    assert not dynamic, f"{dynamic} name {OPS_MODULE} as a string (dynamic import)"


def test_w2b_satellite_never_reads_the_table():
    path, _ = module_file(OPS_MODULE)
    assert path is not None, f"{OPS_MODULE} does not exist"
    for mod in (SATELLITE_MODULE, "cascade.shared.config.satellite"):
        consts = string_constants(mod)
        assert not any("lane_d_ops" in c for c in consts), f"{mod} reads [lane_d_ops]"


def test_w2b_config_package_does_not_reexport_it():
    """``from cascade.shared.config import ...`` (every service, satellites
    included) must not load the ops module."""
    import importlib

    mod = importlib.import_module(OPS_MODULE)
    from .helpers import imported_modules

    pkg = imported_modules("cascade.shared.config", everywhere=True)
    assert OPS_MODULE not in pkg, "cascade/shared/config/__init__.py imports the ops module"
    loader = imported_modules("cascade.shared.config.loader", everywhere=False)
    assert OPS_MODULE not in loader, "loader.py imports the ops module at load time"
    assert mod.__name__ == OPS_MODULE
