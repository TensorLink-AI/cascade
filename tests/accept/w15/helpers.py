"""Shared spec and fixture builders for the W15 prefetch-cache acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6): the W15 row ("Prefetch cache,
LRU with KING_D and house pinned"), §1.2 step 5 ("On reveal, the launch eval
host prefetches the entry into a sha-keyed, resumable cache, using presigned
GETs"), §10 "Data paths" 2 ("presigned GET prefetch at reveal; sha-keyed LRU
with KING_D and the house pinned"), §1.6 ("Byte-based LRU (#467)"), §10.7
(``lane_d_eval_pod``: "disk ≥ cache budget") and §3.2 invariant 1. README.md
states the module contract.

Nothing here imports the W15 module (``cascade.eval.lane_d_cache``) at import
time: on main every test fails at its own first import, inside the test body.
The digest arithmetic is written out here on purpose (the W1.5/W4 manifest
rule: sha256 of the canonical ``{"files": {name: {sha256, size}}}`` JSON), so
the tests do not depend on the code under test to say what an entry is.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

REPO = Path(__file__).resolve().parents[3]

# §1.1 / W4: a lane D checkpoint is exactly these files.
ALLOWED = ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors")
MANIFEST = "manifest.json"
BUCKET = "cascade-lane-d-vault"
PROXY_ENV = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy",
             "ALL_PROXY")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_of(files: dict[str, bytes]) -> dict:
    return {"files": {name: {"sha256": sha256(data), "size": len(data)}
                      for name, data in sorted(files.items())}}


def canonical(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_of(files: dict[str, bytes]) -> str:
    """The entry digest (W1.5/W4): sha256 hex of the canonical manifest."""
    return sha256(canonical(manifest_of(files)))


def entry_files(seed: int, size: int) -> dict[str, bytes]:
    """A deterministic four-file entry of exactly ``size`` bytes in total.
    The cache is content-addressed and does not parse checkpoints, so these
    bytes need not be a model; ``seed`` makes every entry's digest distinct."""
    assert size >= 64
    head = f"seed={seed}\n".encode()
    small = {"config.json": b'{"seed": %d}' % seed,
             "forecast_wrapper.py": head + b"# wrapper\n",
             "model.py": head + b"# model\n"}
    rest = size - sum(len(v) for v in small.values())
    assert rest > 0
    block = hashlib.sha256(b"w15-%d" % seed).digest()
    weights = (block * (rest // len(block) + 1))[:rest]
    files = {**small, "weights.safetensors": weights}
    assert sum(len(v) for v in files.values()) == size
    return files


class Entry:
    """One cache entry the tests can fill: ``files`` (name -> bytes), its
    ``digest``, its byte ``size``, and a ``fill`` callable that writes the
    files into the directory it is given, counting its calls."""

    def __init__(self, seed: int, size: int = 1000):
        self.seed = seed
        self.files = entry_files(seed, size)
        self.digest = digest_of(self.files)
        self.size = size
        self.calls = 0

    def fill(self, directory) -> None:
        self.calls += 1
        d = Path(directory)
        for name, data in self.files.items():
            (d / name).write_bytes(data)


def wrong_fill(files: dict[str, bytes]):
    """A fill that writes ``files`` (bytes that do not hash to the digest asked
    for) and counts its calls in ``fill.calls``."""

    def fill(directory) -> None:
        fill.calls += 1
        for name, data in files.items():
            (Path(directory) / name).write_bytes(data)

    fill.calls = 0
    return fill


def failing_fill(exc: BaseException):
    """A fill that writes one partial file and then raises ``exc``."""

    def fill(directory) -> None:
        fill.calls += 1
        (Path(directory) / "weights.safetensors").write_bytes(b"partial")
        raise exc

    fill.calls = 0
    return fill


def read_tree(directory) -> dict[str, bytes]:
    """Every regular file below ``directory`` (``/``-relative names)."""
    root = Path(directory)
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def entry_dirs(root) -> list[str]:
    """The 64-hex entry directories directly under the cache root, sorted."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and len(p.name) == 64
                  and all(c in "0123456789abcdef" for c in p.name))


def lane_d_cfg(*, from_block: int, until_block: int = 0):
    """A real ``[lane_d]`` config through the repo's parser (bare call, so only
    the from/until ordering is checked)."""
    from cascade.shared.config.lane_d import parse_lane_d

    raw = {"from_block": from_block}
    if until_block:
        raw["until_block"] = until_block
    if from_block:
        raw["share"] = 0.25
    return parse_lane_d(raw)


# ── the external store: presigned GETs on localhost (the only mock) ─────────


class Vault:
    """A minimal presigned-URL object server for the private vault bucket.
    Objects are served at ``/<bucket>/<key>?X-Amz-Signature=...``; every
    request path is recorded in ``requests``."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.requests: list[str] = []
        self._lock = threading.Lock()
        vault = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                return

            def do_GET(self):
                with vault._lock:
                    vault.requests.append(self.path)
                path = urlsplit(self.path).path
                prefix = f"/{BUCKET}/"
                key = unquote(path[len(prefix):]) if path.startswith(prefix) else ""
                data = vault.objects.get(key)
                if data is None:
                    self.send_error(404, "NoSuchKey")
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self, key: str) -> str:
        host, port = self.httpd.server_address[:2]
        return (f"http://{host}:{port}/{BUCKET}/{quote(key)}"
                f"?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=600&X-Amz-Signature=w15")

    def publish(self, hotkey: str, files: dict[str, bytes], *,
                manifest_files: dict[str, bytes] | None = None) -> dict:
        """Upload ``files`` as the miner would and return the box's presigned
        grant. ``manifest_files`` (default ``files``) is what the manifest
        declares, so a test can publish a manifest the objects contradict."""
        declared = files if manifest_files is None else manifest_files
        manifest = manifest_of(declared)
        digest = sha256(canonical(manifest))
        prefix = f"lane-d/{hotkey}/{digest}/"
        for name, data in files.items():
            self.objects[prefix + name] = data
        self.objects[prefix + MANIFEST] = canonical(manifest)
        return {"hotkey": hotkey, "digest": digest, "manifest_url": self.url(prefix + MANIFEST),
                "files": {name: self.url(prefix + name) for name in sorted(declared)}}

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# ── a real lane D fixture checkpoint (W4's T5 fixture shape, no torch) ──────

FIXTURE_CONFIG = {"context_length": 4096, "d_model": 16, "n_heads": 2, "n_layers": 1,
                  "patch_size": 32}
FIXTURE_TENSORS = {
    "blocks.0.attn.weight": ("BF16", (16, 16)),
    "embed.weight": ("BF16", (16, 4)),
    "head.bias": ("BF16", (8,)),
    "head.weight": ("BF16", (8, 16)),
}


def _bf16(n: int) -> bytes:
    import struct

    return b"".join(struct.pack("<f", ((i % 13) - 6) / 8.0)[2:4] for i in range(n))


def fixture_checkpoint() -> dict[str, bytes]:
    """The four allowlisted files of a valid lane D fixture checkpoint: the
    repo's real pyramid code bytes, a small config and a hand-written BF16
    safetensors file (W4 README "T5 cases")."""
    import struct

    trainer = REPO / "cascade" / "trainer"
    header, data = {}, b""
    for name in sorted(FIXTURE_TENSORS):
        dtype, shape = FIXTURE_TENSORS[name]
        n = 1
        for x in shape:
            n *= x
        raw = _bf16(n)
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    weights = struct.pack("<Q", len(blob)) + blob + data
    config = {"base_arch": "cascade_pyramid", "moe": dict(FIXTURE_CONFIG),
              "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    return {"config.json": json.dumps(config, sort_keys=True).encode("utf-8"),
            "forecast_wrapper.py": (trainer / "pyramid_wrapper.py").read_bytes(),
            "model.py": (trainer / "pyramid_model.py").read_bytes(),
            "weights.safetensors": weights}


def fixture_expectation():
    """W4's ``LaneDExpectation`` for :func:`fixture_checkpoint`."""
    from cascade.eval.lane_d_ingest import LaneDExpectation

    files = fixture_checkpoint()
    return LaneDExpectation(
        code={"model.py": files["model.py"], "forecast_wrapper.py": files["forecast_wrapper.py"]},
        config=dict(FIXTURE_CONFIG),
        tensors={k: (dt, tuple(shape)) for k, (dt, shape) in FIXTURE_TENSORS.items()},
        max_weights_bytes=4 * len(files["weights.safetensors"]))
