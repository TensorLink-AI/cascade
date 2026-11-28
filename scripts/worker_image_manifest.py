#!/usr/bin/env python3
"""The worker image's BUILD MANIFEST (W-IMG retro item 8, 2026-11-28).

An image is built by the pipeline (``.github/workflows/publish-worker.yml``)
from ``uv sync --frozen`` (#999) and leaves one manifest behind; a
qualification record (``cascade.shared.image_qualification.validate_build``)
embeds it, so no pin can point at a hand-built image.

Two halves, so the CI runner never pulls the multi-GB image:

``probe`` runs INSIDE the image at build time (``deploy/Dockerfile``, after
the lock-drift guard) and writes ``/root/cascade/.image-probe.json``: the
torch build, its CUDA version and compiled arch list (``sm_NN`` — what the
box checks a pod's ``compute_cap`` against without importing torch), the
Python version, the in-image worker code fingerprint (the exact rule of
:func:`cascade.provision.codeprint.canonical_fingerprint`), the uv.lock
sha256 and whether the lock-drift marker was written. The Dockerfile's
``image-probe`` stage exports that one file; it is the only step that imports
torch, and it never runs on the ops box.

``emit`` runs on the CI runner (stdlib only) after the push: it re-derives the
fingerprint and lock hash from the CHECKOUT the image was built from, refuses
any disagreement with the probe, and writes ``image-manifest.json``::

    {"schema": "cascade-image-build/1", "builder": "publish-worker",
     "image": "ghcr.io/<owner>/cascade-worker@sha256:<64>", "run_id": "...",
     "run_url": "...", "source_commit": "<40 hex>", "source_ref": "...",
     "lock_sha256": "<64>", "lock_frozen": true,
     "worker_code_fingerprint": "<64>", "files": N,
     "torch": "2.4.1+cu124", "cuda": "12.4", "cuda_arch_list": ["sm_50", ...],
     "python": "3.11.x", "built_at": "...Z"}

    python3 scripts/worker_image_manifest.py emit --repo . \\
        --image ghcr.io/tensorlink-ai/cascade-worker@sha256:<64> \\
        --probe image-probe/image-probe.json --source-commit <sha> \\
        --run-id <id> [--run-url URL] [--source-ref REF] --out image-manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

PROBE_SCHEMA = "cascade-image-probe/1"
BUILD_SCHEMA = "cascade-image-build/1"
BUILDER = "publish-worker"
LOCK_MARKER = ".venv/.cascade-lock-verified"
_DIGEST_REF = re.compile(r"^[a-z0-9./_-]+@sha256:[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def code_fingerprint(root: Path) -> tuple[str, int]:
    """``(fingerprint, file count)`` of ``<root>/cascade/**/*.py``
    (``__pycache__`` excluded, paths relative to ``root``), byte for byte the
    canonical rule of ``cascade.provision.codeprint`` — reimplemented here so
    neither half imports the package."""
    pairs = {}
    for p in sorted((root / "cascade").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        pairs[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    lines = "".join(f"{pairs[k]}  {k}\n" for k in sorted(pairs))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest(), len(pairs)


def lock_sha256(root: Path) -> str:
    return hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()


def probe(root: Path) -> dict:
    """The in-image facts (imports torch: INSIDE THE IMAGE BUILD ONLY)."""
    import platform

    import torch  # the image's pinned torch; never on the ops box

    flags = getattr(torch._C, "_cuda_getArchFlags", None)
    arch = sorted(set(str(flags()).split())) if callable(flags) else []
    fp, n = code_fingerprint(root)
    return {"schema": PROBE_SCHEMA, "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda or ""), "cuda_arch_list": arch,
            "python": platform.python_version(), "worker_code_fingerprint": fp, "files": n,
            "lock_sha256": lock_sha256(root), "lock_verified": (root / LOCK_MARKER).is_file()}


def build_manifest(*, repo: Path, image: str, probe_doc: dict, source_commit: str,
                   run_id: str, run_url: str = "", source_ref: str = "",
                   built_at: str | None = None) -> tuple[dict, list[str]]:
    """``(manifest, problems)``: the manifest of ``image`` built from
    ``repo`` at ``source_commit``; any problem means the image is NOT the
    checkout's lock-frozen build and no manifest may be published."""
    bad: list[str] = []
    image = str(image).strip()
    if not _DIGEST_REF.match(image):
        bad.append(f"image {image!r} is not a lowercase repo@sha256:<64 hex> ref")
    if not _HEX40.match(str(source_commit)):
        bad.append(f"source commit {source_commit!r} is not 40 hex")
    if not str(run_id).strip():
        bad.append("run id missing")
    if probe_doc.get("schema") != PROBE_SCHEMA:
        bad.append(f"probe schema {probe_doc.get('schema')!r} != {PROBE_SCHEMA!r}")
    fp, n = code_fingerprint(repo)
    lock = lock_sha256(repo)
    if probe_doc.get("worker_code_fingerprint") != fp:
        bad.append(f"in-image worker code fingerprint {str(probe_doc.get('worker_code_fingerprint'))[:12]}… "
                   f"!= the checkout's {fp[:12]}… (the image is not this source)")
    if probe_doc.get("lock_sha256") != lock:
        bad.append("in-image uv.lock differs from the checkout's")
    if probe_doc.get("lock_verified") is not True:
        bad.append("the lock-drift marker is absent: the venv was not verified against uv.lock")
    arch = probe_doc.get("cuda_arch_list")
    if not isinstance(arch, list) or not arch:
        bad.append("the torch build reports no CUDA arch list")
    for key in ("torch", "cuda", "python"):
        if not str(probe_doc.get(key, "") or "").strip():
            bad.append(f"probe {key} missing")
    manifest = {
        "schema": BUILD_SCHEMA, "builder": BUILDER, "image": image,
        "run_id": str(run_id), "run_url": str(run_url), "source_commit": str(source_commit),
        "source_ref": str(source_ref), "lock_sha256": lock,
        "lock_frozen": probe_doc.get("lock_verified") is True,
        "worker_code_fingerprint": fp, "files": n,
        "torch": str(probe_doc.get("torch", "")), "cuda": str(probe_doc.get("cuda", "")),
        "cuda_arch_list": list(arch) if isinstance(arch, list) else [],
        "python": str(probe_doc.get("python", "")),
        "built_at": built_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return manifest, bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="inside the image build: write the probe JSON")
    p.add_argument("--root", type=Path, default=Path("/root/cascade"))
    p.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("emit", help="on the CI runner: write image-manifest.json")
    e.add_argument("--repo", type=Path, default=Path("."))
    e.add_argument("--image", required=True)
    e.add_argument("--probe", type=Path, required=True)
    e.add_argument("--source-commit", required=True)
    e.add_argument("--source-ref", default="")
    e.add_argument("--run-id", required=True)
    e.add_argument("--run-url", default="")
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "probe":
        doc = probe(args.root)
        args.out.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n")
        print(json.dumps(doc, sort_keys=True))
        return 0
    manifest, bad = build_manifest(
        repo=args.repo.resolve(), image=args.image,
        probe_doc=json.loads(args.probe.read_text(encoding="utf-8")),
        source_commit=args.source_commit, run_id=args.run_id, run_url=args.run_url,
        source_ref=args.source_ref)
    if bad:
        print("IMAGE MANIFEST REFUSED:", file=sys.stderr)
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
