"""Give CI an isolated CA store with the four revoked incident roots removed.

No download, host trust-store mutation or verification bypass is involved.
Print environment assignments for GitHub Actions' GITHUB_ENV file.
"""
from __future__ import annotations

import argparse
import hashlib
import ssl
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_tls_trust import AFFECTED_ROOTS, PEM_CERT  # noqa: E402


def prepare(source: Path, directory: Path) -> dict[str, str]:
    text = source.read_text(encoding="ascii")
    if "-----BEGIN TRUSTED CERTIFICATE-----" in text:
        raise ValueError("CI bootstrap requires an ordinary PEM CA bundle")
    clean = []
    for certificate in PEM_CERT.findall(text):
        fingerprint = hashlib.sha256(ssl.PEM_cert_to_DER_cert(certificate)).hexdigest().upper()
        if fingerprint not in AFFECTED_ROOTS:
            clean.append(certificate)
    if not clean:
        raise ValueError("CI bootstrap found no unaffected CA certificates")
    bundle = "\n".join(clean) + "\n"
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cadata=bundle)  # Refuse malformed certificates.
    if not context.get_ca_certs():
        raise ValueError("CI bootstrap found no usable CA certificates")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "bundle.pem"
    path.write_text(bundle, encoding="ascii")
    env = {name: str(path) for name in (
        "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "AWS_CA_BUNDLE", "GIT_SSL_CAINFO")}
    env["SSL_CERT_DIR"] = str(directory)
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source", type=Path, default=Path("/etc/ssl/certs/ca-certificates.crt"))
    args = parser.parse_args(argv)
    for name, value in prepare(args.source, args.directory).items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
