"""Check the CA stores used by a worker, including lazy OpenSSL CA directories.

Run with the application's Python after installation. Before installation,
--system-only checks OS trust without requiring certifi. No network is used.
Fingerprints follow the eMudhra response, not certificate display names:
https://github.com/astral-sh/uv/security/advisories/GHSA-hq85-5cpj-mr62
https://github.com/certifi/python-certifi/releases/tag/2026.10.19
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import ssl
from pathlib import Path

MIN_CERTIFI = (2026, 10, 19)
AFFECTED_ROOTS = frozenset(
    {
        "7DD78D5F4F13459A83DFF9ABBBA62EDBAF6F2D102BF257FD712F4D9F2746ED8D",
        "F2EDAAFF3D9B701FB4D14C9F285F8C0BAD90FC3BCC1FA07D47F4718EDABA34A5",
        "D7A7A0FB5D7E2731D771E9484EBCDEF71D5F0C3E0A2948782BC83EE0EA699EF4",
        "86A1ECBA089C4A8D3BBE2734C612BA341D813E043CF9E8A862CD5C57A36BBE6B",
    }
)
PEM_CERT = re.compile(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.DOTALL)


def fingerprints(context: ssl.SSLContext) -> set[str]:
    return {
        hashlib.sha256(cert).hexdigest().upper() for cert in context.get_ca_certs(binary_form=True)
    }


def file_fingerprints(path: Path) -> set[str]:
    text = path.read_text(encoding="ascii", errors="replace")
    found = {
        hashlib.sha256(ssl.PEM_cert_to_DER_cert(cert)).hexdigest().upper()
        for cert in PEM_CERT.findall(text)
    }
    if "-----BEGIN TRUSTED CERTIFICATE-----" in text:
        # OpenSSL's extended PEM encoding has auxiliary trust data. Let OpenSSL
        # decode it so a lazy CA directory cannot conceal an affected anchor.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cafile=str(path))
        found.update(fingerprints(context))
    return found


def path_fingerprints(path: Path) -> set[str]:
    if not path.is_dir():
        return file_fingerprints(path)
    # OpenSSL loads hashed directory entries only when a chain needs them;
    # SSLContext.get_ca_certs() alone therefore misses these certificates.
    found: set[str] = set()
    seen: set[Path] = set()
    for entry in path.iterdir():
        resolved = entry.resolve()
        if resolved in seen or not entry.is_file():
            continue
        seen.add(resolved)
        found.update(file_fingerprints(entry))
    return found


def audit(*, system_only: bool = False) -> dict:
    stores: dict[str, set[str]] = {}
    errors: list[str] = []

    def check_path(label: str, path: str) -> None:
        try:
            values = path_fingerprints(Path(path))
            if not values:
                errors.append(f"{label}: no readable CA certificates")
            stores[label] = values
        except (OSError, ValueError) as exc:
            errors.append(f"{label}: cannot inspect CA store ({type(exc).__name__})")

    try:
        stores["ssl.default"] = fingerprints(ssl.create_default_context())
    except (OSError, ValueError) as exc:
        errors.append(f"ssl.default: cannot load trust ({type(exc).__name__})")
    defaults = ssl.get_default_verify_paths()
    for label, path in (("ssl.cafile", defaults.cafile), ("ssl.capath", defaults.capath)):
        if path:
            for part in path.split(os.pathsep):
                check_path(label + ":" + str(len(stores)), part)
    # Inspect custom stores even if this client normally ignores that variable.
    # Never include environment values in diagnostics (they may contain secrets).
    for name in (
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "AWS_CA_BUNDLE",
    ):
        if name in os.environ:
            for i, path in enumerate(os.environ[name].split(os.pathsep)):
                if not path:
                    errors.append(f"{name}: empty CA override")
                else:
                    check_path(f"{name}:{i}", path)

    if not system_only:
        try:
            version = importlib.metadata.version("certifi")
            parts = tuple(int(part) for part in version.split("."))
            if parts < MIN_CERTIFI:
                errors.append("certifi: requires version >=2026.10.19")
        except (importlib.metadata.PackageNotFoundError, ValueError):
            errors.append("certifi: missing or unrecognised version")
        for name, attr in (
            ("certifi", "where"),
            ("requests.certs", "where"),
            ("botocore.httpsession", "get_cert_path"),
            ("pip._vendor.certifi", "where"),
        ):
            try:
                module = importlib.import_module(name)
            except ModuleNotFoundError as exc:
                if name == "certifi" or not (name == exc.name or name.startswith(exc.name + ".")):
                    errors.append(f"{name}: cannot import CA provider")
                continue
            try:
                bundle = (
                    getattr(module, attr)(True)
                    if attr == "get_cert_path"
                    else getattr(module, attr)()
                )
                check_path(name, bundle)
            except (AttributeError, TypeError, OSError, ValueError):
                errors.append(f"{name}: cannot discover CA store")

    rows = [
        {
            "store": name,
            "certificates": len(values),
            "affected_roots": sorted(values & AFFECTED_ROOTS),
        }
        for name, values in stores.items()
    ]
    if not any(row["certificates"] for row in rows if row["store"].startswith("ssl.")):
        errors.append("ssl: no inspectable default CA certificates")
    return {
        "ok": not errors and not any(row["affected_roots"] for row in rows),
        "stores": rows,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-only", action="store_true")
    args = parser.parse_args()
    result = audit(system_only=args.system_only)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
