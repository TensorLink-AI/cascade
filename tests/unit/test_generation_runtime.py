"""CPU generator dependencies must be installed and importable before miner code."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from scripts.check_generation_runtime import LIBRARIES

ROOT = Path(__file__).resolve().parents[2]


def test_generation_extra_declares_the_missing_generator_libraries():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = project["project"]["optional-dependencies"]["generation"]
    for requirement in ("numba==0.66.0", "scikit-learn==1.9.0", "statsmodels==0.14.6", "gpytorch==1.15.2"):
        assert requirement in requirements
    allowed = tomllib.loads((ROOT / "chain.toml").read_text())["dependencies"]["allowed"]
    assert set(allowed) <= set(LIBRARIES)


def test_bootstrap_import_smoke_reports_the_frozen_runtime():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/check_generation_runtime.py")],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    assert proc.returncode == 0, proc.stderr
    versions = json.loads(proc.stdout)["generation_runtime"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    for name in LIBRARIES:
        expected = {p["version"] for p in lock["package"] if p["name"] == name}
        assert versions[name] in expected, (name, versions[name], expected)
