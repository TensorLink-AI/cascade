#!/usr/bin/env python3
"""Derive ``[round] worker_code_fingerprint`` for a pinned worker image.

    python scripts/worker_code_fingerprint.py ghcr.io/tensorlink-ai/cascade-worker@sha256:<hex>

Walks the image's registry layers (newest first, CUDA base skipped by size),
hashes every ``/root/cascade/cascade/**/*.py`` and prints the canonical
fingerprint (cascade.provision.codeprint.canonical_fingerprint) plus the file
count. Paste the value into chain.toml next to ``funded_pod_image`` on every
image re-pin; the provisioner gate and the funded rent path refuse pods whose
files hash differently (a stale image served under the pinned tag).

Auth: ``GHCR_TOKEN`` (a PAT with read:packages, used as ``tensorlink-dev:<pat>``
for the token exchange) from the environment / .env; anonymous works for
public images. Compare with a live pod:

    ssh <pod> 'cd /root/cascade && find cascade -type f -name "*.py" -not -path "*/__pycache__/*" | xargs sha256sum' \
      | python -c 'import sys; from cascade.provision.codeprint import parse_sha256sum, canonical_fingerprint; print(canonical_fingerprint(parse_sha256sum(sys.stdin.read(), workdir="/root/cascade")))'
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str]) -> int:
    if len(argv) != 2 or "@sha256:" not in argv[1]:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        from cascade.shared.env import load_env_files
        load_env_files()
    except Exception:  # noqa: BLE001 — env files are optional here
        pass
    from cascade.provision.codeprint import registry_code_fingerprint

    pat = os.environ.get("GHCR_TOKEN", "")
    token = f"{os.environ.get('GHCR_USER', 'tensorlink-dev')}:{pat}" if pat else ""
    fp, n = registry_code_fingerprint(argv[1], token=token)
    print(f"worker_code_fingerprint = \"{fp}\"   # {n} files, {argv[1]}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
