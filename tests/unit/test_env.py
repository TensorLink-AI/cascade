"""load_env_files — auto-load a gitignored .env into the process environment at
CLI startup, without clobbering already-exported vars."""

from __future__ import annotations

import pytest

from cascade.shared.env import load_env_files

pytest.importorskip("dotenv")   # python-dotenv is a core dep; skip if absent


def _write_env(tmp_path, body: str):
    (tmp_path / ".env").write_text(body)


def test_loads_vars_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BACKUP_S3_ACCESS_KEY", raising=False)
    _write_env(tmp_path, "BACKUP_S3_ACCESS_KEY=from-dotenv\n")
    load_env_files()
    import os
    assert os.environ["BACKUP_S3_ACCESS_KEY"] == "from-dotenv"


def test_exported_var_wins_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BACKUP_S3_SECRET_KEY", "exported")
    _write_env(tmp_path, "BACKUP_S3_SECRET_KEY=from-dotenv\n")
    load_env_files()
    import os
    assert os.environ["BACKUP_S3_SECRET_KEY"] == "exported"   # override=False


def test_disabled_by_env_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BACKUP_S3_ACCESS_KEY", raising=False)
    monkeypatch.setenv("CASCADE_NO_DOTENV", "1")
    _write_env(tmp_path, "BACKUP_S3_ACCESS_KEY=from-dotenv\n")
    load_env_files()
    import os
    assert os.environ.get("BACKUP_S3_ACCESS_KEY") is None


def test_no_dotenv_file_is_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # empty dir, no .env
    load_env_files()                     # must not raise
