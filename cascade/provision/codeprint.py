"""Worker-code fingerprint — the byte-level identity of the ``cascade`` package
on a pod, computed from the pod's ACTUAL files.

Why this exists (2026-09-12): a Lium template names the image by repo + tag
and the pinned digest only rides the container env
(``CASCADE_TRAIN_IMAGE_DIGEST``). A host that already holds an OLDER image
under that tag starts *that* container — and still answers the digest gate
with the value we injected, because the gate reads the env we set. Two Lium
hosts (91.224.44.222/.223) ran the Aug-27 ``worker-v0.7.0`` code under a
``worker-v0.8.0`` env: every vault-ref leg dispatched there sat on a ref the
old worker could not resolve until the 1800 s stream-stall fired, and the
miner was blamed ("generator_stalled") and burned.

The fingerprint cannot be spoofed by launch injection: it is a sha256 over
the sorted ``"<sha256hex>  <path>\\n"`` lines of every ``*.py`` under the
pod's ``<workdir>/cascade`` (``__pycache__`` excluded, paths relative to the
workdir). The PIN is derived the same way from the pinned image's registry
layers (``scripts/worker_code_fingerprint.py``) and lives in chain.toml
``[round] worker_code_fingerprint`` next to the digest pin.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable

PACKAGE_DIR = "cascade"
CHECK_TIMEOUT_SECONDS = 120


def canonical_fingerprint(pairs: Iterable[tuple[str, str]]) -> str:
    """sha256 hex over ``"<sha256hex>  <path>\\n"`` lines sorted by path.

    ``pairs`` are ``(path, sha256hex)`` with ``path`` relative to the workdir
    (``cascade/trainer/loop.py``). Paths are normalised (a leading ``./`` is
    dropped) so a remote ``find`` and a registry walk agree byte-for-byte.
    """
    norm = {}
    for path, digest in pairs:
        p = str(path)
        while p.startswith("./"):
            p = p[2:]
        norm[p] = str(digest).strip().lower()
    lines = "".join(f"{norm[p]}  {p}\n" for p in sorted(norm))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def find_argv(workdir: str) -> list[str]:
    """Remote argv listing the package's ``*.py`` files (one per line)."""
    return ["find", f"{workdir.rstrip('/')}/{PACKAGE_DIR}", "-type", "f",
            "-name", "*.py", "-not", "-path", "*/__pycache__/*"]


def parse_sha256sum(stdout: str, *, workdir: str) -> list[tuple[str, str]]:
    """``sha256sum`` lines → ``(relative path, hex)`` pairs.

    Accepts both the two-space and the ``*``-prefixed (binary) forms and
    strips the workdir prefix so the pairs are workdir-relative.
    """
    root = workdir.rstrip("/") + "/"
    out: list[tuple[str, str]] = []
    for line in (stdout or "").splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        digest, _, path = line.partition("  ")
        if not path:
            digest, _, path = line.partition(" ")
        path = path.lstrip("*").strip()
        digest = digest.strip()
        if len(digest) != 64 or not path:
            continue
        if path.startswith(root):
            path = path[len(root):]
        out.append((path, digest))
    return out


def remote_code_fingerprint(run_ssh: Callable[[list[str]], object], *,
                            workdir: str) -> tuple[str, int, str]:
    """``(fingerprint, file_count, error)`` for the package on a pod.

    Two single-purpose remote commands (no pipeline, no remote quoting to
    trust): ``find`` for the file list, ``sha256sum`` over exactly those
    paths. ``error`` is non-empty (and fingerprint empty) when either step
    fails or lists nothing — an image without the package is a mismatch, not
    a pass.
    """
    proc = run_ssh(find_argv(workdir))
    if getattr(proc, "returncode", 1) != 0:
        return "", 0, f"find failed (rc={getattr(proc, 'returncode', '?')}): " \
                      f"{(getattr(proc, 'stderr', '') or '')[-200:]}"
    paths = sorted(p.strip() for p in (getattr(proc, "stdout", "") or "").splitlines()
                   if p.strip())
    if not paths:
        return "", 0, f"no {PACKAGE_DIR}/**/*.py under {workdir}"
    proc = run_ssh(["sha256sum", *paths])
    if getattr(proc, "returncode", 1) != 0:
        return "", 0, f"sha256sum failed (rc={getattr(proc, 'returncode', '?')}): " \
                      f"{(getattr(proc, 'stderr', '') or '')[-200:]}"
    pairs = parse_sha256sum(getattr(proc, "stdout", "") or "", workdir=workdir)
    if len(pairs) != len(paths):
        return "", len(pairs), f"hashed {len(pairs)} of {len(paths)} files"
    return canonical_fingerprint(pairs), len(pairs), ""


def registry_code_fingerprint(image_ref: str, *, token: str = "",
                              registry: str = "https://ghcr.io",
                              max_layer_bytes: int = 400_000_000,
                              workdir: str = "/root/cascade") -> tuple[str, int]:
    """``(fingerprint, file_count)`` of ``<repo>@<digest>`` computed from the
    image's registry layers (newest layer wins per path, like the union
    filesystem). Skips layers above ``max_layer_bytes`` (the CUDA base) — the
    ``COPY . /root/cascade`` layer is small. ``token`` is a GHCR bearer or a
    ``user:pat`` pair for the token exchange; anonymous works for public
    images. Network — release tooling only, never on a round's path.
    """
    import requests

    repo, sep, digest = image_ref.partition("@")
    if not sep:
        raise ValueError(f"image ref must be <repo>@sha256:<hex>: {image_ref!r}")
    host, _, name = repo.partition("/")
    if host != registry.split("://", 1)[-1]:
        raise ValueError(f"image {repo!r} is not on {registry}")
    if ":" in token:
        user, _, pat = token.partition(":")
        bearer = requests.get(f"{registry}/token", params={"scope": f"repository:{name}:pull"},
                              auth=(user, pat), timeout=30).json().get("token", "")
    else:
        bearer = token or requests.get(f"{registry}/token",
                                       params={"scope": f"repository:{name}:pull"},
                                       timeout=30).json().get("token", "")
    hdr = {"Authorization": f"Bearer {bearer}",
           "Accept": ", ".join(["application/vnd.oci.image.index.v1+json",
                                "application/vnd.oci.image.manifest.v1+json",
                                "application/vnd.docker.distribution.manifest.list.v2+json",
                                "application/vnd.docker.distribution.manifest.v2+json"])}
    base = f"{registry}/v2/{name}"
    m = requests.get(f"{base}/manifests/{digest}", headers=hdr, timeout=60).json()
    if "manifests" in m:
        sub = [x for x in m["manifests"]
               if x.get("platform", {}).get("architecture") == "amd64"] or m["manifests"]
        m = requests.get(f"{base}/manifests/{sub[0]['digest']}", headers=hdr, timeout=60).json()
    prefix = workdir.strip("/") + f"/{PACKAGE_DIR}/"
    files: dict[str, str] = {}
    for layer in reversed(m["layers"]):
        if int(layer.get("size", 0)) > max_layer_bytes:
            continue
        blob = requests.get(f"{base}/blobs/{layer['digest']}", headers=hdr, timeout=600).content
        _walk_layer(blob, prefix=prefix, strip=len(workdir.strip("/")) + 1, files=files)
    return canonical_fingerprint(files.items()), len(files)


def _walk_layer(blob: bytes, *, prefix: str, strip: int, files: dict[str, str]) -> None:
    """Add the layer's package ``*.py`` files to ``files`` (first writer wins —
    callers walk newest layer first)."""
    import gzip
    import io
    import tarfile

    def _members(fileobj):
        with tarfile.open(fileobj=fileobj, mode="r|") as tf:
            for ti in tf:
                nm = ti.name
                while nm.startswith("./"):
                    nm = nm[2:]
                if (ti.isfile() and nm.startswith(prefix) and nm.endswith(".py")
                        and "__pycache__" not in nm):
                    rel = nm[strip:]
                    if rel not in files:
                        files[rel] = hashlib.sha256(tf.extractfile(ti).read()).hexdigest()

    try:
        with gzip.GzipFile(fileobj=io.BytesIO(blob)) as gz:
            _members(gz)
    except (OSError, EOFError, tarfile.TarError):  # uncompressed layer
        _members(io.BytesIO(blob))
