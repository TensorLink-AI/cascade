"""W10: `cascade lane-d verify` runs W4's ingest guards on a local entry
directory and never imports or executes anything in it (plan §1.1, §3.2
invariant 3). Contract: README.md "verify"."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from .helpers import REPO, cli_env, repo_code, run_cli, write_checkpoint

_BLOCK_TORCH = "raise ImportError('W10 test: torch is blocked for the allowlist check')\n"
_MARKER_PY = "import os\nos.mkdir({marker!r})\n"


def _sandbox(tmp_path):
    d = tmp_path / "w10_sandbox" / "torch"
    d.mkdir(parents=True, exist_ok=True)
    (d / "__init__.py").write_text(_BLOCK_TORCH)
    return d.parent


def _verify_without_torch(tmp_path, entry_dir):
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("HIPPIUS_") or k.startswith("CASCADE_"))}
    env["PYTHONPATH"] = os.pathsep.join([str(_sandbox(tmp_path)), str(REPO)])
    env["CASCADE_NO_DOTENV"] = "1"
    code = ("import sys\nfrom cascade.miner.cli import main\n"
            f"sys.exit(main(['lane-d', 'verify', {str(entry_dir)!r}, '--json']))\n")
    proc = subprocess.run([sys.executable, "-P", "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120, check=False)
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    doc = json.loads(lines[-1]) if lines else {}
    return proc, doc


@pytest.mark.parametrize("case", ["extra_py", "nested_py", "pickle_pt", "pickle_bin"])
def test_verify_refuses_a_disallowed_file_without_torch_or_execution(tmp_path, case):
    marker = tmp_path / f"executed-{case}"
    payload = _MARKER_PY.format(marker=str(marker)).encode()
    extra = {"extra_py": {"evil.py": payload}, "nested_py": {"pkg/__init__.py": payload},
             "pickle_pt": {"model.pt": b"\x80\x04K\x01."},
             "pickle_bin": {"pytorch_model.bin": b"\x80\x04K\x01."}}[case]
    entry = write_checkpoint(tmp_path / "entry", extra=extra)
    proc, doc = _verify_without_torch(tmp_path, entry)
    assert (REPO / "cascade" / "miner" / "lane_d.py").is_file(), \
        "cascade/miner/lane_d.py does not exist"
    assert proc.returncode == 1, (proc.returncode, proc.stderr[-2000:])
    assert (doc.get("error") or {}).get("code") == "file_not_allowed", doc
    assert not marker.exists(), "verify executed a file from the entry"


def test_verify_of_a_missing_directory_is_a_usage_error(tmp_path, monkeypatch, capsys, caplog):
    cli_env(monkeypatch, tmp_path)
    res = run_cli(capsys, caplog, "lane-d", "verify", str(tmp_path / "nope"))
    assert res.rc in (1, 2) and res.code, res


@pytest.mark.torch
def test_verify_refuses_edited_code_with_code_mismatch(tmp_path, monkeypatch, capsys, caplog):
    cli_env(monkeypatch, tmp_path)
    entry = write_checkpoint(tmp_path / "entry")
    (entry / "model.py").write_bytes(b"# edited\n" + repo_code()["model.py"])
    res = run_cli(capsys, caplog, "lane-d", "verify", str(entry))
    assert res.rc == 1 and res.code == "code_mismatch", res


@pytest.mark.torch
def test_verify_codes_are_w4s_ingest_codes(tmp_path, monkeypatch, capsys, caplog):
    from cascade.eval.lane_d_ingest import IngestCode

    cli_env(monkeypatch, tmp_path)
    entry = write_checkpoint(tmp_path / "entry")
    (entry / "config.json").write_text("{not json")
    res = run_cli(capsys, caplog, "lane-d", "verify", str(entry))
    assert res.rc == 1, res
    assert res.code in {c.value for c in IngestCode}, res
