"""WB19: the tool as an operator runs it, `python -P scripts/laned_wb19_measure.py`.

Subprocess runs with a scrubbed env. The tool's prod guard reads this host's
real ``/``, so on the prod box only the prod-refusal tests run and the rest
skip (CI runners carry no prod marker). No torch, no GPU. Contract:
README.md "`run`" and "Never on the prod box".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from .helpers import (
    IMAGE_ENV,
    REPO,
    SECRET,
    SECRET_ENV,
    TOOL,
    dry_run_argv,
    host_is_prod,
    pinned_image,
    refusal_code,
)

not_on_prod = pytest.mark.skipif(host_is_prod(), reason="this host is the prod box: the "
                                 "tool refuses here by design (prod_host); CI runs these")


def _env(tmp_path: Path, **extra) -> dict:
    bindir = tmp_path / "emptybin"
    bindir.mkdir(exist_ok=True)
    env = {"PATH": str(bindir), "HOME": str(tmp_path), "PYTHONPATH": str(REPO),
           IMAGE_ENV: pinned_image()}
    env.update(extra)
    return env


def _cli(argv, env, cwd):
    assert TOOL.is_file(), "scripts/laned_wb19_measure.py is missing (WB19 builds it)"
    return subprocess.run([sys.executable, "-P", str(TOOL), *[str(a) for a in argv]],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


def test_module_import_is_torch_free(tmp_path):
    assert TOOL.is_file(), "scripts/laned_wb19_measure.py is missing (WB19 builds it)"
    code = ("import importlib.util, sys; "
            f"s = importlib.util.spec_from_file_location('m', {str(TOOL)!r}); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "print('torch' in sys.modules)")
    p = subprocess.run([sys.executable, "-P", "-c", code], cwd=tmp_path, env=_env(tmp_path),
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "False", "torch is imported only inside the real leg path"


def test_cli_refuses_prod_role(tmp_path):
    out = tmp_path / "r.json"
    p = _cli(dry_run_argv(tmp_path, "8xH100", out=out), _env(tmp_path, CASCADE_HOST_ROLE="prod"),
             tmp_path)
    assert p.returncode == 2 and refusal_code(p.stderr) == "prod_host", p.stderr
    assert not out.exists()


@pytest.mark.skipif(not host_is_prod(), reason="only meaningful on the prod box")
def test_cli_refuses_on_this_prod_box(tmp_path):
    out = tmp_path / "r.json"
    p = _cli(dry_run_argv(tmp_path, "8xH100", out=out), _env(tmp_path), tmp_path)
    assert p.returncode == 2 and refusal_code(p.stderr) == "prod_host", p.stderr
    assert not out.exists()


@not_on_prod
def test_cli_dry_run_never_imports_torch(tmp_path):
    """A planted `torch` that raises on import: the dry run still succeeds."""
    fake = tmp_path / "planted"
    (fake / "torch").mkdir(parents=True)
    (fake / "torch" / "__init__.py").write_text("raise ImportError('planted torch: WB19 "
                                                "dry-run imported torch')\n")
    env = _env(tmp_path, PYTHONPATH=f"{fake}{os.pathsep}{REPO}")
    out = tmp_path / "r.json"
    p = _cli(dry_run_argv(tmp_path, "8xH100", out=out), env, tmp_path)
    assert p.returncode == 0, p.stderr
    assert json.loads(out.read_text())["source"] == "dry_run"


@not_on_prod
def test_cli_record_is_hashseed_independent(tmp_path):
    outs = []
    for seed in ("1", "4242"):
        d = tmp_path / seed
        d.mkdir()
        out = d / "r.json"
        p = _cli(dry_run_argv(d, "4xH100", out=out), _env(d, PYTHONHASHSEED=seed), d)
        assert p.returncode == 0, p.stderr
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]


@not_on_prod
def test_cli_never_prints_secrets(tmp_path):
    secrets = {k: f"{SECRET}-{k}" for k in SECRET_ENV}
    out = tmp_path / "r.json"
    p1 = _cli(dry_run_argv(tmp_path, "8xH200", out=out), _env(tmp_path, **secrets), tmp_path)
    assert p1.returncode == 0, p1.stderr
    p2 = _cli(dry_run_argv(tmp_path, "L40S", out=tmp_path / "x.json"),
              _env(tmp_path, **secrets), tmp_path)
    assert p2.returncode == 2 and refusal_code(p2.stderr) == "gpu_not_candidate", p2.stderr
    blob = p1.stdout + p1.stderr + p2.stdout + p2.stderr + out.read_text()
    assert SECRET not in blob


def test_cli_secrets_stay_out_of_a_prod_refusal(tmp_path):
    secrets = {k: f"{SECRET}-{k}" for k in SECRET_ENV}
    p = _cli(dry_run_argv(tmp_path, "8xH100", out=tmp_path / "r.json"),
             _env(tmp_path, CASCADE_HOST_ROLE="prod", **secrets), tmp_path)
    assert p.returncode == 2 and refusal_code(p.stderr) == "prod_host", p.stderr
    assert SECRET not in p.stdout + p.stderr
