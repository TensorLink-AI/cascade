"""Public incident certificates; no private keys or live network requests.

Chains: https://emudhra.we-have-a-problem.com/ (downloaded 2026-10-23).
Use a fixed verification date so certificate expiry cannot hide a regression.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import ssl
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import certifi
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/tls/emudhra"
spec = importlib.util.spec_from_file_location(
    "check_tls_trust", ROOT / "scripts/check_tls_trust.py"
)
trust = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trust)


def test_locks_require_patched_certifi():
    for project in (ROOT, ROOT / "benchmarks"):
        metadata = tomllib.loads((project / "pyproject.toml").read_text())
        assert "certifi>=2026.10.19" in metadata["project"]["dependencies"]
        lock = tomllib.loads((project / "uv.lock").read_text())
        package = next(p for p in lock["package"] if p["name"] == "certifi")
        assert tuple(map(int, package["version"].split("."))) >= trust.MIN_CERTIFI
        assert (
            package["wheels"][0]["hash"]
            == "sha256:42d35340cc5b9ba78e818ddbff0c5b1c2b779de448a340fd1b989f5fa8670e1b"
        )


def test_public_affected_chains_fail_with_patched_store():
    openssl = shutil.which("openssl")
    if not openssl:
        pytest.skip("openssl required for public chain regression")
    timestamp = int(datetime(2026, 10, 23, 12, tzinfo=UTC).timestamp())

    def verify(bundle, chain):
        return subprocess.run(
            [
                openssl,
                "verify",
                "-attime",
                str(timestamp),
                "-purpose",
                "sslserver",
                "-verify_hostname",
                chain.stem,
                "-no-CApath",
                "-no-CAstore",
                "-CAfile",
                str(bundle),
                "-untrusted",
                str(chain),
                str(chain),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    chains = sorted(FIXTURES.glob("*.we-have-a-problem.com.pem"))
    assert len(chains) == 14
    # Positive vulnerability control: the affected root still validates a real
    # compromised chain. A broken fixture/command must not masquerade as safety.
    canary = FIXTURES / "dv-g3b-noct-emudhra.we-have-a-problem.com.pem"
    control = verify(FIXTURES / "affected-roots.pem", canary)
    assert control.returncode == 0, control.stderr
    for chain in chains:
        result = verify(certifi.where(), chain)
        assert result.returncode != 0, result.stdout
        assert "unable to get local issuer certificate" in result.stderr, result.stderr


@pytest.fixture
def isolated_stores(monkeypatch, tmp_path):
    for name in (
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "AWS_CA_BUNDLE",
    ):
        monkeypatch.delenv(name, raising=False)
    clean = tmp_path / "clean.pem"
    clean.write_text(trust.PEM_CERT.findall(Path(certifi.where()).read_text())[0] + "\n")
    monkeypatch.setattr(
        trust.ssl,
        "get_default_verify_paths",
        lambda: SimpleNamespace(cafile=str(clean), capath=None),
    )
    monkeypatch.setattr(
        trust.ssl, "create_default_context", lambda: ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    )
    return clean


def test_clean_store_passes(isolated_stores):
    assert trust.audit(system_only=True)["ok"]


def test_lazy_capath_is_checked(monkeypatch, isolated_stores, tmp_path):
    directory = tmp_path / "certs"
    directory.mkdir()
    (directory / "12345678.0").symlink_to(FIXTURES / "affected-roots.pem")
    monkeypatch.setattr(
        trust.ssl,
        "get_default_verify_paths",
        lambda: SimpleNamespace(cafile=str(isolated_stores), capath=str(directory)),
    )
    result = trust.audit(system_only=True)
    assert not result["ok"]
    assert any(set(row["affected_roots"]) == trust.AFFECTED_ROOTS for row in result["stores"])


@pytest.mark.parametrize(
    "name",
    ["SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "AWS_CA_BUNDLE"],
)
def test_unsafe_override_rejected(name, monkeypatch, isolated_stores):
    monkeypatch.setenv(name, str(FIXTURES / "affected-roots.pem"))
    assert not trust.audit(system_only=True)["ok"]


@pytest.mark.parametrize("value", ["", "/missing/private-path-that-must-not-be-logged"])
def test_uninspectable_override_fails_without_disclosing_value(value, monkeypatch, isolated_stores):
    monkeypatch.setenv("AWS_CA_BUNDLE", value)
    result = trust.audit(system_only=True)
    assert not result["ok"]
    if value:
        assert value not in str(result)


def test_extended_openssl_pem_cannot_hide_affected_root(isolated_stores, tmp_path):
    openssl = shutil.which("openssl")
    if not openssl:
        pytest.skip("openssl required for extended PEM fixture")
    result = subprocess.run(
        [
            openssl,
            "x509",
            "-in",
            str(FIXTURES / "affected-roots.pem"),
            "-addtrust",
            "serverAuth",
            "-trustout",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    bundle = tmp_path / "mixed.pem"
    bundle.write_text(isolated_stores.read_text() + result.stdout)
    assert trust.path_fingerprints(bundle) & trust.AFFECTED_ROOTS


def test_old_certifi_version_fails_even_with_clean_bundle(monkeypatch, isolated_stores):
    monkeypatch.setattr(trust.importlib.metadata, "version", lambda name: "2026.6.17")
    assert "certifi: requires version >=2026.10.19" in trust.audit()["errors"]


def test_missing_certifi_fails(monkeypatch, isolated_stores):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(trust.importlib.metadata, "version", missing)
    assert not trust.audit()["ok"]


@pytest.fixture
def bootstrap_environment(tmp_path):
    """Execute the real shell control flow with all remote effects isolated."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    source = tmp_path / "worker with spaces"
    source.mkdir()
    (source / ".venv/bin").mkdir(parents=True)
    (source / "benchmarks/.venv/bin").mkdir(parents=True)
    log = tmp_path / "calls"
    mock = """#!PYTHON
import os, pathlib, shutil, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['MOCK_LOG'], 'a') as out:
    out.write(name + ' ' + ' '.join(args) + '\\n')
if name == 'uv':
    if args == ['--version']:
        print('uv ' + os.environ.get('MOCK_UV_VERSION', '0.13.4'))
    elif args and args[0] == 'run':
        sys.exit(subprocess.run([os.environ['POD_WORKDIR'] + '/.venv/bin/python', 'scripts/check_tls_trust.py', '--system-only']).returncode)
    elif 'benchmarks' in args:
        sys.exit(int(os.environ.get('MOCK_BENCH_EXIT', '0')))
elif name == 'scp':
    destination = pathlib.Path(args[-1].split(':', 1)[1])
    shutil.copyfile(args[-2], destination)
    if os.environ.get('MOCK_CORRUPT_UV'):
        destination.write_bytes(b'corrupt transfer')
elif name == 'ssh':
    sys.exit(subprocess.run(args[-1], shell=True).returncode)
elif name == 'python':
    if 'check_tls_trust.py' in ' '.join(args):
        sys.exit(int(os.environ.get('MOCK_TRUST_EXIT', '0')))
""".replace("#!PYTHON", "#!" + sys.executable)
    for name in ("uv", "ssh", "scp", "rsync", "sysctl"):
        p = binaries / name
        p.write_text(mock)
        p.chmod(0o755)
    for p in (source / ".venv/bin/python", source / "benchmarks/.venv/bin/python"):
        p.write_text(mock)
        p.chmod(0o755)
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "UV_NATIVE_TLS",
            "UV_SYSTEM_CERTS",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "UV_INSECURE_HOST",
        }
    }
    env.update(
        PATH=str(binaries) + os.pathsep + env["PATH"],
        MOCK_LOG=str(log),
        POD_IP="127.0.0.1",
        POD_USER="fixture",
        POD_PORT="22",
        POD_KEY="fixture-only",
        POD_WORKDIR=str(source),
        POD_STAGE="eval",
        CASCADE_SRC_ROOT=str(source),
        CASCADE_UV_BIN=str(binaries / "uv"),
    )

    def run(**overrides):
        result = subprocess.run(
            ["bash", str(ROOT / "deploy/pod_bootstrap.sh")],
            env=env | overrides,
            capture_output=True,
            text=True,
            check=False,
        )
        return result, log.read_text() if log.exists() else ""

    return run


def test_bootstrap_checks_trust_before_bench_and_without_cached_data(bootstrap_environment):
    result, log = bootstrap_environment()
    assert result.returncode == 0, result.stderr
    assert log.index("python scripts/check_tls_trust.py") < log.index(
        "uv sync --project benchmarks"
    )
    assert (
        sum(line.startswith("python scripts/check_tls_trust.py") for line in log.splitlines()) == 3
    )
    assert "--extra time --extra fev" in log


@pytest.mark.parametrize("version", ["0.10.4", "0.13.3", "0.13.4rc1", "invalid"])
def test_bootstrap_rejects_old_uv_before_remote_actions(bootstrap_environment, version):
    result, log = bootstrap_environment(MOCK_UV_VERSION=version)
    assert result.returncode != 0
    assert "requires" in result.stderr
    assert "rsync" not in log and "ssh " not in log


def test_bootstrap_rejects_corrupt_installer(bootstrap_environment):
    result, log = bootstrap_environment(MOCK_CORRUPT_UV="1")
    assert result.returncode != 0
    assert "uv sync" not in log


def test_bootstrap_propagates_benchmark_failure(bootstrap_environment):
    result, log = bootstrap_environment(MOCK_BENCH_EXIT="7")
    assert result.returncode == 7
    assert (
        sum(line.startswith("python scripts/check_tls_trust.py") for line in log.splitlines()) == 2
    )


def test_bootstrap_stops_on_unsafe_trust(bootstrap_environment):
    result, log = bootstrap_environment(MOCK_TRUST_EXIT="1")
    assert result.returncode == 1
    assert "--project benchmarks" not in log


def test_bootstrap_refuses_uv_trust_override(bootstrap_environment):
    result, log = bootstrap_environment(UV_SYSTEM_CERTS="true")
    assert result.returncode != 0
    assert "uv sync" not in log
