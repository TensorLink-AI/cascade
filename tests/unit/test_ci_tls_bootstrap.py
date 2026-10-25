"""Hosted-runner bootstrap removes only known bad roots before network fetches."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import certifi
import pytest

from scripts.check_tls_trust import AFFECTED_ROOTS, file_fingerprints
from scripts.prepare_ci_tls import prepare

ROOT = Path(__file__).resolve().parents[2]
BAD_ROOTS = ROOT / "tests/fixtures/tls/emudhra/affected-roots.pem"


def test_bootstrap_preserves_other_roots_and_passes_the_unchanged_guard(tmp_path):
    source = tmp_path / "host-bundle.pem"
    original = Path(certifi.where()).read_text() + "\n" + BAD_ROOTS.read_text()
    source.write_text(original)
    env = prepare(source, tmp_path / "ci-store")
    cleaned = file_fingerprints(Path(env["SSL_CERT_FILE"]))
    assert cleaned == file_fingerprints(source) - AFFECTED_ROOTS
    assert cleaned and not (cleaned & AFFECTED_ROOTS)
    assert source.read_text() == original  # The host store stays untouched.
    result = subprocess.run([sys.executable, str(ROOT / "scripts/check_tls_trust.py"), "--system-only"],
        env={**os.environ, **env}, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"ok": true' in result.stdout
    # Git, requests and OpenSSL all receive the same filtered bundle; a lazy
    # OpenSSL CA directory cannot reintroduce the host's revoked anchors.
    assert env["GIT_SSL_CAINFO"] == env["REQUESTS_CA_BUNDLE"] == env["SSL_CERT_FILE"]
    assert list(Path(env["SSL_CERT_DIR"]).iterdir()) == [Path(env["SSL_CERT_FILE"])]


@pytest.mark.parametrize("text", ["", "not a PEM bundle", BAD_ROOTS.read_text()])
def test_bootstrap_refuses_to_produce_an_empty_trust_store(tmp_path, text):
    source = tmp_path / "source.pem"
    source.write_text(text)
    with pytest.raises(ValueError, match="no unaffected"):
        prepare(source, tmp_path / "ci-store")
    assert not (tmp_path / "ci-store").exists()
