"""Shared spec and fixture builders for the W4 ingest-guard acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6): the W4 row, §1.1 (exact file
allowlist, BF16 only, lane A's F32 guard unchanged), §1.2 (presigned GETs
only, no credentials on pods), §3.2 invariant 3, §4 T5/T19/T28, §10.4/§10.5
(handshake first; eval host restarted before the validator). README.md states
the module contract.

Nothing here imports a W4 module (a PATHS module) at import time: on main
every test fails at its own first import, inside the test body.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import os
import pickle
import shutil
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

REPO = Path(__file__).resolve().parents[3]

# §1.1 / W4: a lane D checkpoint is exactly these files, nothing else.
ALLOWED = ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors")
MANIFEST = "manifest.json"
VAULT_BUCKET = "cascade-lane-d-vault"

# Credential env names that must never be needed (or leak) on the presigned
# path. The values set by the tests are sentinels, never real secrets.
CREDENTIAL_ENV = (
    "HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY", "HIPPIUS_ACCESS_KEY_ID",
    "HIPPIUS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN", "HF_TOKEN",
)
PROXY_ENV = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy",
             "ALL_PROXY")

# The fixture model the non-torch guards are checked against (README
# "LaneDExpectation"): the repo's real pyramid code bytes, a small model
# config (the ``"moe"`` object of config.json) and a small BF16 tensor spec.
FIXTURE_CONFIG = {"context_length": 4096, "d_model": 16, "n_heads": 2, "n_layers": 1,
                  "patch_size": 32}
FIXTURE_TENSORS = {
    "blocks.0.attn.weight": ("BF16", (16, 16)),
    "embed.weight": ("BF16", (16, 4)),
    "head.bias": ("BF16", (8,)),
    "head.weight": ("BF16", (8, 16)),
}
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8}
BF16_NAN = b"\xc0\x7f"   # little-endian 0x7FC0
BF16_INF = b"\x80\x7f"   # little-endian 0x7F80


def repo_code() -> dict[str, bytes]:
    """The exact code bytes a lane D (cascade_pyramid) checkpoint must carry."""
    trainer = REPO / "cascade" / "trainer"
    return {"model.py": (trainer / "pyramid_model.py").read_bytes(),
            "forecast_wrapper.py": (trainer / "pyramid_wrapper.py").read_bytes()}


def _numel(shape) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n


def tensor_bytes(dtype: str, shape, *, poison: bytes | None = None) -> bytes:
    """Deterministic finite data for one tensor; ``poison`` (BF16 NaN/Inf
    bytes) replaces its first element."""
    n = _numel(shape)
    values = [((i % 13) - 6) / 8.0 for i in range(n)]
    if dtype == "BF16":
        raw = b"".join(struct.pack("<f", v)[2:4] for v in values)
    elif dtype == "F32":
        raw = struct.pack(f"<{n}f", *values)
    else:
        raise ValueError(dtype)
    if poison is not None:
        raw = poison + raw[len(poison):]
    return raw


def safetensors_bytes(tensors: dict[str, tuple[str, tuple, bytes]], *, tail: bytes = b"",
                      offsets: dict[str, list[int]] | None = None) -> bytes:
    """A safetensors file written directly: u64 LE header length, JSON header
    ``{name: {dtype, shape, data_offsets}}``, then the data."""
    header, data = {}, b""
    for name in sorted(tensors):
        dtype, shape, raw = tensors[name]
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    for name, offs in (offsets or {}).items():
        header[name]["data_offsets"] = list(offs)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    return struct.pack("<Q", len(blob)) + blob + data + tail


def read_header(path: Path) -> dict:
    """The test's own safetensors header reader (no torch, no safetensors)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def good_tensors(spec=None, *, dtype: str | None = None, poison: bytes | None = None):
    spec = FIXTURE_TENSORS if spec is None else spec
    out = {}
    for i, (name, (dt, shape)) in enumerate(sorted(spec.items())):
        dt = dtype or dt
        out[name] = (dt, shape, tensor_bytes(dt, shape, poison=poison if i == 0 else None))
    return out


def good_weights() -> bytes:
    return safetensors_bytes(good_tensors())


def config_json(config=None) -> bytes:
    body = {"base_arch": "cascade_pyramid", "moe": dict(FIXTURE_CONFIG if config is None else config),
            "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    return json.dumps(body, sort_keys=True).encode("utf-8")


def max_weights_bytes() -> int:
    """The fixture byte budget: generous (4× the good BF16 file, so the F32
    variant is refused for its dtype, not its size)."""
    return 4 * len(good_weights())


def expectation_kwargs() -> dict:
    """Keyword arguments of ``LaneDExpectation`` for the fixture model."""
    return {"code": repo_code(), "config": dict(FIXTURE_CONFIG),
            "tensors": {k: (dt, tuple(shape)) for k, (dt, shape) in FIXTURE_TENSORS.items()},
            "max_weights_bytes": max_weights_bytes()}


def write_good_checkpoint(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in repo_code().items():
        (directory / name).write_bytes(data)
    (directory / "config.json").write_bytes(config_json())
    (directory / "weights.safetensors").write_bytes(good_weights())
    return directory


# ── planted payloads: each one would create its marker if it ever executed ──


def planted_python(marker: Path) -> bytes:
    return f"import os\nos.mkdir({str(marker)!r})\n".encode()


class _Plant:
    """Pickles to ``os.mkdir(marker)``: unpickling it executes miner bytes."""

    def __init__(self, marker: Path):
        self.marker = str(marker)

    def __reduce__(self):
        return (os.mkdir, (self.marker,))


def planted_pickle(marker: Path) -> bytes:
    return pickle.dumps(_Plant(marker), protocol=4)


# ── the T5 malformed set (README table "T5 cases") ──────────────────────────
#
# name -> (exact IngestCode value, fetch policy). Fetch policy:
#   "manifest_only": refused from the manifest alone; no checkpoint object is
#                    fetched at all (allowlist refusals);
#   "no_weights":    weights.safetensors is never fetched (size budget);
#   None:            no constraint.

T5_CASES = {
    "extra_py": ("file_not_allowed", "manifest_only"),
    "nested_py": ("file_not_allowed", "manifest_only"),
    "torch_pt": ("file_not_allowed", "manifest_only"),
    "pickle_bin": ("file_not_allowed", "manifest_only"),
    "pickle_pkl": ("file_not_allowed", "manifest_only"),
    "edited_model_py": ("code_mismatch", None),
    "edited_wrapper_py": ("code_mismatch", None),
    "wrong_config": ("config_mismatch", None),
    "unparseable_config": ("config_mismatch", None),
    "pickle_as_weights": ("bad_header", None),
    "bad_data_offsets": ("bad_header", None),
    "f32_dtype": ("dtype_mismatch", None),
    "wrong_shape": ("shape_mismatch", None),
    "extra_tensor": ("tensor_set_mismatch", None),
    "missing_tensor": ("tensor_set_mismatch", None),
    "oversize": ("oversize", "no_weights"),
}
# Need the finite check, which may load tensors with torch: CI only.
T5_TORCH_CASES = {
    "nan_weights": ("non_finite", None),
    "inf_weights": ("non_finite", None),
}


def build_case(case: str, root: Path) -> tuple[Path, Path]:
    """A checkpoint directory for one T5 case and the marker path that must
    never appear. ``root`` is a fresh directory."""
    ckpt = write_good_checkpoint(root / "ckpt")
    marker = root / "markers" / case
    marker.parent.mkdir(parents=True, exist_ok=True)
    code = repo_code()
    w = ckpt / "weights.safetensors"
    if case == "extra_py":
        (ckpt / "evil.py").write_bytes(planted_python(marker))
    elif case == "nested_py":
        (ckpt / "pkg").mkdir()
        (ckpt / "pkg" / "__init__.py").write_bytes(planted_python(marker))
    elif case == "torch_pt":
        (ckpt / "model.pt").write_bytes(planted_pickle(marker))
    elif case == "pickle_bin":
        (ckpt / "pytorch_model.bin").write_bytes(planted_pickle(marker))
    elif case == "pickle_pkl":
        (ckpt / "state.pkl").write_bytes(planted_pickle(marker))
    elif case == "edited_model_py":
        (ckpt / "model.py").write_bytes(planted_python(marker) + code["model.py"])
    elif case == "edited_wrapper_py":
        (ckpt / "forecast_wrapper.py").write_bytes(planted_python(marker)
                                                   + code["forecast_wrapper.py"])
    elif case == "wrong_config":
        (ckpt / "config.json").write_bytes(config_json({**FIXTURE_CONFIG, "d_model": 32}))
    elif case == "unparseable_config":
        (ckpt / "config.json").write_bytes(b'{"moe": {"d_model": 16,')
    elif case == "pickle_as_weights":
        w.write_bytes(planted_pickle(marker))
    elif case == "bad_data_offsets":
        tensors = good_tensors()
        start = 0
        for name in sorted(tensors):
            if name == "head.bias":
                break
            start += len(tensors[name][2])
        w.write_bytes(safetensors_bytes(tensors, offsets={"head.bias": [start, start + 4]}))
    elif case == "f32_dtype":
        w.write_bytes(safetensors_bytes(good_tensors(dtype="F32")))
    elif case == "wrong_shape":
        tensors = good_tensors()
        dt, _, raw = tensors["head.weight"]
        tensors["head.weight"] = (dt, (16, 8), raw)
        w.write_bytes(safetensors_bytes(tensors))
    elif case == "extra_tensor":
        tensors = good_tensors()
        tensors["extra.weight"] = ("BF16", (2,), tensor_bytes("BF16", (2,)))
        w.write_bytes(safetensors_bytes(tensors))
    elif case == "missing_tensor":
        tensors = good_tensors()
        del tensors["head.bias"]
        w.write_bytes(safetensors_bytes(tensors))
    elif case == "oversize":
        w.write_bytes(safetensors_bytes(good_tensors(), tail=b"\0" * (max_weights_bytes() + 1)))
    elif case == "nan_weights":
        w.write_bytes(safetensors_bytes(good_tensors(poison=BF16_NAN)))
    elif case == "inf_weights":
        w.write_bytes(safetensors_bytes(good_tensors(poison=BF16_INF)))
    else:
        raise KeyError(case)
    return ckpt, marker


def counting_storage(root: Path):
    """The real filesystem adapter, instrumented to record every ``get``."""
    from cascade.lane_d.storage import FilesystemStorage

    class CountingStorage(FilesystemStorage):
        def __init__(self, r):
            super().__init__(r)
            self.gets: list[str] = []

        def get(self, key):
            self.gets.append(key)
            return super().get(key)

    return CountingStorage(root)


def foreign_modules(tmp: Path) -> list[str]:
    """Imported modules whose source lies under ``tmp`` (miner bytes)."""
    import sys

    base = str(Path(tmp).resolve())
    out = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None) or ""
        if f and str(Path(f).resolve()).startswith(base):
            out.append(name)
    return sorted(out)


def dir_is_clean(dest: Path) -> bool:
    """A refused ingest leaves nothing behind in ``dest``."""
    return not dest.exists() or not any(dest.iterdir())


# ── the external store: a presigned-URL object server on localhost ──────────


class VaultServer:
    """Simulates the private vault bucket's presigned GETs (the only mocked
    service). Objects are served at
    ``/<bucket>/<key>?X-Amz-Expires=600&X-Amz-Signature=<sig>``; a missing or
    wrong signature gets 403. The signing secret lives only in this object.
    Every request (path, query, headers) and the bytes written per key are
    recorded."""

    CHUNK = 1 << 20

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.declared: dict[str, int] = {}    # key -> Content-Length of zero bytes
        self.endless: dict[str, int] = {}     # key -> cap of a close-delimited stream
        self.requests: list[dict] = []
        self.sent: dict[str, int] = {}
        self._secret = os.urandom(32)
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                return

            def do_GET(self):
                parts = urlsplit(self.path)
                query = parse_qs(parts.query)
                with server._lock:
                    server.requests.append({"path": self.path, "query": query,
                                            "headers": {k.lower(): v for k, v in self.headers.items()}})
                prefix = f"/{VAULT_BUCKET}/"
                key = unquote(parts.path[len(prefix):]) if parts.path.startswith(prefix) else ""
                sig = (query.get("X-Amz-Signature") or [""])[0]
                if not key or not hmac.compare_digest(sig, server.signature(key)):
                    self.send_error(403, "SignatureDoesNotMatch")
                    return
                try:
                    if key in server.endless:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.end_headers()
                        server._stream(self, key, server.endless[key])
                    elif key in server.declared:
                        self.send_response(200)
                        self.send_header("Content-Length", str(server.declared[key]))
                        self.end_headers()
                        server._stream(self, key, server.declared[key])
                    elif key in server.objects:
                        data = server.objects[key]
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        server._count(key, len(data))
                    else:
                        self.send_error(404, "NoSuchKey")
                except (BrokenPipeError, ConnectionResetError):
                    return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def _count(self, key: str, n: int) -> None:
        with self._lock:
            self.sent[key] = self.sent.get(key, 0) + n

    def _stream(self, handler, key: str, total: int) -> None:
        chunk = b"\0" * self.CHUNK
        left = total
        while left > 0:
            part = chunk[:min(left, self.CHUNK)]
            try:
                handler.wfile.write(part)
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            self._count(key, len(part))
            left -= len(part)

    def signature(self, key: str) -> str:
        return hmac.new(self._secret, key.encode("utf-8"), hashlib.sha256).hexdigest()

    def url(self, key: str, *, valid: bool = True) -> str:
        sig = self.signature(key) if valid else self.signature(key + "#forged")
        host, port = self.httpd.server_address[:2]
        return (f"http://{host}:{port}/{VAULT_BUCKET}/{quote(key)}"
                f"?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=600&X-Amz-Signature={sig}")

    def got(self) -> list[str]:
        """Object keys requested so far, in order."""
        prefix = f"/{VAULT_BUCKET}/"
        with self._lock:
            return [unquote(urlsplit(r["path"]).path[len(prefix):]) for r in self.requests]

    def publish(self, hotkey: str, directory: Path) -> dict:
        """Upload ``directory`` as the miner would (manifest + files under
        ``lane-d/<hotkey>/<digest>/``) and return the box's presigned grant."""
        from cascade.lane_d.intake import canonical_manifest, checkpoint_manifest, manifest_digest

        manifest = checkpoint_manifest(directory)
        digest = manifest_digest(manifest)
        prefix = f"lane-d/{hotkey}/{digest}/"
        for name in manifest["files"]:
            self.objects[prefix + name] = (Path(directory) / name).read_bytes()
        self.objects[prefix + MANIFEST] = canonical_manifest(manifest)
        return {"hotkey": hotkey, "digest": digest, "manifest_url": self.url(prefix + MANIFEST),
                "files": {name: self.url(prefix + name) for name in sorted(manifest["files"])}}

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# ── torch-only builders (CI): the repo's real smallest pyramid model ────────


def pyramid_contract():
    """The smallest pyramid size (``cascade-moe-v2-2m``) via the repo's own
    ``scripts/pyramid_config.py`` (loaded by path), as W1.5 does."""
    from cascade.shared.config import parse_chain_config

    spec = importlib.util.spec_from_file_location(
        "w4_pyramid_config", REPO / "scripts" / "pyramid_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return parse_chain_config(mod.local_config("2m")).training


def tiny_checkpoint(directory: Path, contract, *, seed: int, bf16: bool,
                    poison: float | None = None) -> Path:
    """A random-init 2m pyramid checkpoint written by the real trainer code;
    ``poison`` (nan/inf) overwrites the first element of the first floating
    tensor."""
    import torch

    from cascade.trainer.pyramid_model import CascadeMoEConfig, CascadeMoEModel
    from cascade.trainer.pyramid_trainer import save_checkpoint

    config = CascadeMoEConfig.from_contract(contract)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = CascadeMoEModel(config)
    state, poisoned = {}, False
    for name, tensor in model.state_dict().items():
        t = tensor.detach().clone()
        if bf16 and t.is_floating_point():
            t = t.to(torch.bfloat16)
        if poison is not None and not poisoned and t.is_floating_point():
            t.view(-1)[0] = poison
            poisoned = True
        state[name] = t.contiguous()
    save_checkpoint(directory, state, config, contract)
    return directory


def copy_tree(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst
